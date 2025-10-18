// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_cardiovascular0d_respiratory_syspulperiphcirculation.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_utils_function_of_time.hpp"

#include <iostream>

FOUR_C_NAMESPACE_OPEN



/*----------------------------------------------------------------------*
 |  ctor (public)                                              mhv 10/13|
 *----------------------------------------------------------------------*/
Utils::CardiovascularRespiratory0DSysPulPeriphCirculation::
    CardiovascularRespiratory0DSysPulPeriphCirculation(
        std::shared_ptr<Core::FE::Discretization> discr, const std::string& conditionname,
        std::vector<int>& curID)
    : Cardiovascular0D(discr, conditionname, curID)
{
  Teuchos::ParameterList artvensyspulpar =
      Global::Problem::instance()->cardiovascular0_d_structural_params().sublist(
          "SYS-PUL CIRCULATION PARAMETERS");

  num_dof_cardio_ = 34;
  num_dof_respir_ = 48;

  // set all 0D model parameters
  r_arvalve_max_l_ = artvensyspulpar.get("R_arvalve_max_l", 0.0);
  r_arvalve_min_l_ = artvensyspulpar.get("R_arvalve_min_l", 0.0);
  r_atvalve_max_l_ = artvensyspulpar.get("R_atvalve_max_l", 0.0);
  r_atvalve_min_l_ = artvensyspulpar.get("R_atvalve_min_l", 0.0);
  r_arvalve_max_r_ = artvensyspulpar.get("R_arvalve_max_r", 0.0);
  r_arvalve_min_r_ = artvensyspulpar.get("R_arvalve_min_r", 0.0);
  r_atvalve_max_r_ = artvensyspulpar.get("R_atvalve_max_r", 0.0);
  r_atvalve_min_r_ = artvensyspulpar.get("R_atvalve_min_r", 0.0);
  atrium_act_curve_l_ = artvensyspulpar.get(
      "Atrium_act_curve_l", -1);  // left atrial activation curve (ONLY for ATRIUM_MODEL "0D"!)
  atrium_act_curve_r_ = artvensyspulpar.get(
      "Atrium_act_curve_r", -1);  // right atrial activation curve (ONLY for ATRIUM_MODEL "0D"!)
  ventricle_act_curve_l_ = artvensyspulpar.get("Ventricle_act_curve_l",
      -1);  // left ventricular activation curve (ONLY for VENTRICLE_MODEL "0D"!)
  ventricle_act_curve_r_ = artvensyspulpar.get("Ventricle_act_curve_r",
      -1);  // right ventricular activation curve (ONLY for VENTRICLE_MODEL "0D"!)
  atrium_prescr_e_curve_l_ = artvensyspulpar.get("Atrium_prescr_E_curve_l",
      -1);  // left atrial elastance prescription curve (ONLY for ATRIUM_MODEL "prescribed"!)
  atrium_prescr_e_curve_r_ = artvensyspulpar.get("Atrium_prescr_E_curve_r",
      -1);  // right atrial elastance prescription curve (ONLY for ATRIUM_MODEL "prescribed"!)
  ventricle_prescr_e_curve_l_ = artvensyspulpar.get(
      "Ventricle_prescr_E_curve_l", -1);  // left ventricular elastance prescription curve (ONLY for
                                          // VENTRICLE_MODEL "prescribed"!)
  ventricle_prescr_e_curve_r_ = artvensyspulpar.get(
      "Ventricle_prescr_E_curve_r", -1);  // right ventricular elastance prescription curve (ONLY
                                          // for VENTRICLE_MODEL "prescribed"!)
  e_at_max_l_ = artvensyspulpar.get("E_at_max_l", 0.0);
  e_at_min_l_ = artvensyspulpar.get("E_at_min_l", 0.0);
  e_at_max_r_ = artvensyspulpar.get("E_at_max_r", 0.0);
  e_at_min_r_ = artvensyspulpar.get("E_at_min_r", 0.0);
  e_v_max_l_ = artvensyspulpar.get("E_v_max_l", 0.0);
  e_v_min_l_ = artvensyspulpar.get("E_v_min_l", 0.0);
  e_v_max_r_ = artvensyspulpar.get("E_v_max_r", 0.0);
  e_v_min_r_ = artvensyspulpar.get("E_v_min_r", 0.0);
  c_ar_sys_ = artvensyspulpar.get("C_ar_sys", 0.0);
  r_ar_sys_ = artvensyspulpar.get("R_ar_sys", 0.0);
  l_ar_sys_ = artvensyspulpar.get("L_ar_sys", 0.0);
  z_ar_sys_ = artvensyspulpar.get("Z_ar_sys", 0.0);

  // peripheral arterial compliances and resistances
  c_arspl_sys_ = artvensyspulpar.get("C_arspl_sys", 0.0);
  r_arspl_sys_ = artvensyspulpar.get("R_arspl_sys", 0.0);
  c_arespl_sys_ = artvensyspulpar.get("C_arespl_sys", 0.0);
  r_arespl_sys_ = artvensyspulpar.get("R_arespl_sys", 0.0);
  c_armsc_sys_ = artvensyspulpar.get("C_armsc_sys", 0.0);
  r_armsc_sys_ = artvensyspulpar.get("R_armsc_sys", 0.0);
  c_arcer_sys_ = artvensyspulpar.get("C_arcer_sys", 0.0);
  r_arcer_sys_ = artvensyspulpar.get("R_arcer_sys", 0.0);
  c_arcor_sys_ = artvensyspulpar.get("C_arcor_sys", 0.0);
  r_arcor_sys_ = artvensyspulpar.get("R_arcor_sys", 0.0);
  // peripheral venous compliances and resistances
  c_venspl_sys_ = artvensyspulpar.get("C_venspl_sys", 0.0);
  r_venspl_sys_ = artvensyspulpar.get("R_venspl_sys", 0.0);
  c_venespl_sys_ = artvensyspulpar.get("C_venespl_sys", 0.0);
  r_venespl_sys_ = artvensyspulpar.get("R_venespl_sys", 0.0);
  c_venmsc_sys_ = artvensyspulpar.get("C_venmsc_sys", 0.0);
  r_venmsc_sys_ = artvensyspulpar.get("R_venmsc_sys", 0.0);
  c_vencer_sys_ = artvensyspulpar.get("C_vencer_sys", 0.0);
  r_vencer_sys_ = artvensyspulpar.get("R_vencer_sys", 0.0);
  c_vencor_sys_ = artvensyspulpar.get("C_vencor_sys", 0.0);
  r_vencor_sys_ = artvensyspulpar.get("R_vencor_sys", 0.0);

  c_ar_pul_ = artvensyspulpar.get("C_ar_pul", 0.0);
  r_ar_pul_ = artvensyspulpar.get("R_ar_pul", 0.0);
  l_ar_pul_ = artvensyspulpar.get("L_ar_pul", 0.0);
  z_ar_pul_ = artvensyspulpar.get("Z_ar_pul", 0.0);
  // pulmonary capillary compliance and resistance
  c_cap_pul_ = artvensyspulpar.get("C_cap_pul", 0.0);
  r_cap_pul_ = artvensyspulpar.get("R_cap_pul", 0.0);

  c_ven_sys_ = artvensyspulpar.get("C_ven_sys", 0.0);
  r_ven_sys_ = artvensyspulpar.get("R_ven_sys", 0.0);
  l_ven_sys_ = artvensyspulpar.get("L_ven_sys", 0.0);
  c_ven_pul_ = artvensyspulpar.get("C_ven_pul", 0.0);
  r_ven_pul_ = artvensyspulpar.get("R_ven_pul", 0.0);
  l_ven_pul_ = artvensyspulpar.get("L_ven_pul", 0.0);

  // unstressed volumes
  v_v_l_u_ = artvensyspulpar.get("V_v_l_u", 1.0);
  v_at_l_u_ = artvensyspulpar.get("V_at_l_u", 1.0);
  v_ar_sys_u_ = artvensyspulpar.get("V_ar_sys_u", 1.0);

  v_arspl_sys_u_ = artvensyspulpar.get("V_venspl_sys_u", 1.0);
  v_arespl_sys_u_ = artvensyspulpar.get("V_venespl_sys_u", 1.0);
  v_armsc_sys_u_ = artvensyspulpar.get("V_venmsc_sys_u", 1.0);
  v_arcer_sys_u_ = artvensyspulpar.get("V_vencer_sys_u", 1.0);
  v_arcor_sys_u_ = artvensyspulpar.get("V_vencor_sys_u", 1.0);
  v_venspl_sys_u_ = artvensyspulpar.get("V_venspl_sys_u", 1.0);
  v_venespl_sys_u_ = artvensyspulpar.get("V_venespl_sys_u", 1.0);
  v_venmsc_sys_u_ = artvensyspulpar.get("V_venmsc_sys_u", 1.0);
  v_vencer_sys_u_ = artvensyspulpar.get("V_vencer_sys_u", 1.0);
  v_vencor_sys_u_ = artvensyspulpar.get("V_vencor_sys_u", 1.0);

  v_ven_sys_u_ = artvensyspulpar.get("V_ven_sys_u", 1.0);
  v_v_r_u_ = artvensyspulpar.get("V_v_r_u", 1.0);
  v_at_r_u_ = artvensyspulpar.get("V_at_r_u", 1.0);
  v_ar_pul_u_ = artvensyspulpar.get("V_ar_pul_u", 1.0);
  v_cap_pul_u_ = artvensyspulpar.get("V_cap_pul_u", 1.0);
  v_ven_pul_u_ = artvensyspulpar.get("V_ven_pul_u", 1.0);


  // now set the parameters for the 0D respiratory model
  Teuchos::ParameterList respirpar =
      Global::Problem::instance()->cardiovascular0_d_structural_params().sublist(
          "RESPIRATORY PARAMETERS");

  // set number of degrees of freedom
  switch (respiratory_model_)
  {
    case Inpar::Cardiovascular0D::resp_none:
      num_dof_ = num_dof_cardio_;
      break;
    case Inpar::Cardiovascular0D::resp_standard:
      num_dof_ = num_dof_cardio_ + num_dof_respir_;
      break;
    default:
      FOUR_C_THROW("Undefined respiratory_model!");
      break;
  }


  l_alv_ = respirpar.get("L_alv", 0.0);
  r_alv_ = respirpar.get("R_alv", 0.0);
  e_alv_ = respirpar.get("E_alv", 0.0);

  u_t_curve_ = respirpar.get("U_t_curve", -1);
  u_m_ = respirpar.get("U_m", 0.0);


  v_lung_tidal_ = respirpar.get("V_lung_tidal",
      400.0);  // tidal volume (the total volume of inspired air, in a single breath)
  v_lung_dead_ = respirpar.get("V_lung_dead", 150.0);  // dead space volume
  v_lung_u_ = respirpar.get("V_lung_u", 0.0);  // unstressed lung volume (volume of the lung when it
                                               // is fully collapsed outside the body)

  f_c_o2_ext_ = respirpar.get("fCO2_ext", 0.03);
  f_o2_ext_ = respirpar.get("fO2_ext", 0.21);

  // should be 22.4 liters per mol !
  // however we specify it as an input parameter since its decimal power depends on the system of
  // units your whole model is specified in! i.e. if you have kg - mm - s - mmol, it's 22.4e3 mm^3 /
  // mmol
  v_m_gas_ = respirpar.get("V_m_gas", 22.4e3);  // molar volume of an ideal gas

  // should be 47.1 mmHg = 6.279485 kPa !
  // however we specify it as an input parameter since its decimal power depends on the system of
  // units your whole model is specified in! i.e. if you have kg - mm - s - mmol, it's 6.279485 kPa
  p_vap_water_37_ =
      respirpar.get("p_vap_water_37", 6.279485);  // vapor pressure of water at 37  degrees celsius


  kappa_c_o2_ =
      respirpar.get("kappa_CO2", 0.0);  // diffusion coefficient for CO2 across the hemato-alveolar
                                        // membrane, in molar value / (time * pressure)
  kappa_o2_ =
      respirpar.get("kappa_O2", 0.0);  // diffusion coefficient for CO2 across the hemato-alveolar
                                       // membrane, in molar value / (time * pressure)

  alpha_c_o2_ = respirpar.get(
      "alpha_CO2", 0.0);  // CO2 solubility constant, in molar value / (volume * pressure)
  alpha_o2_ = respirpar.get(
      "alpha_O2", 0.0);  // O2 solubility constant, in molar value / (volume * pressure)

  c_hb_ = respirpar.get(
      "c_Hb", 9.3e-6);  // hemoglobin concentration of the blood, in molar value / volume (default:
                        // Christiansen (1996), p. 92, unit: mmol/mm^3)

  m_c_o2_arspl_ = respirpar.get("M_CO2_arspl", 0.0);  // splanchnic metabolic rate of CO2 production
  m_o2_arspl_ = respirpar.get("M_O2_arspl", 0.0);     // splanchnic metabolic rate of O2 consumption
  m_c_o2_arespl_ =
      respirpar.get("M_CO2_arespl", 0.0);  // extra-splanchnic metabolic rate of CO2 production
  m_o2_arespl_ =
      respirpar.get("M_O2_arespl", 0.0);  // extra-splanchnic metabolic rate of O2 consumption
  m_c_o2_armsc_ = respirpar.get("M_CO2_armsc", 0.0);  // muscular metabolic rate of CO2 production
  m_o2_armsc_ = respirpar.get("M_O2_armsc", 0.0);     // muscular metabolic rate of O2 consumption
  m_c_o2_arcer_ = respirpar.get("M_CO2_arcer", 0.0);  // cerebral metabolic rate of CO2 production
  m_o2_arcer_ = respirpar.get("M_O2_arcer", 0.0);     // cerebral metabolic rate of O2 consumption
  m_c_o2_arcor_ = respirpar.get("M_CO2_arcor", 0.0);  // coronary metabolic rate of CO2 production
  m_o2_arcor_ = respirpar.get("M_O2_arcor", 0.0);     // coronary metabolic rate of O2 consumption

  v_tissspl_ = respirpar.get("V_tissspl", 1.0);
  v_tissespl_ = respirpar.get("V_tissespl", 1.0);
  v_tissmsc_ = respirpar.get("V_tissmsc", 1.0);
  v_tisscer_ = respirpar.get("V_tisscer", 1.0);
  v_tisscor_ = respirpar.get("V_tisscor", 1.0);
}



/*-----------------------------------------------------------------------*
 |(private)                                                    mhv 02/15 |
 |Evaluate method for a closed-loop 0D vascular model                    |
 |(Hirschvogel, Bassilious, Jagschies, Wildhirt, Gee, "A monolithic 3D-0D|
 |coupled closed-loop model of the heart and the vascular system:        |
 |Experiment-based parameter estimation for patient-specific cardiac     |
 |mechanics", IJNMBE, 2016)                                              |
 *-----------------------------------------------------------------------*/
void Utils::CardiovascularRespiratory0DSysPulPeriphCirculation::evaluate(
    Teuchos::ParameterList& params, std::shared_ptr<Core::LinAlg::SparseMatrix> sysmat1,
    std::shared_ptr<Core::LinAlg::SparseOperator> sysmat2,
    std::shared_ptr<Core::LinAlg::SparseOperator> sysmat3,
    std::shared_ptr<Core::LinAlg::Vector<double>> sysvec1,
    std::shared_ptr<Core::LinAlg::Vector<double>> sysvec2,
    std::shared_ptr<Core::LinAlg::Vector<double>> sysvec3,
    const std::shared_ptr<Core::LinAlg::Vector<double>> sysvec4,
    std::shared_ptr<Core::LinAlg::Vector<double>> sysvec5)
{
  if (!actdisc_->filled()) FOUR_C_THROW("fill_complete() was not called");
  if (!actdisc_->have_dofs()) FOUR_C_THROW("assign_degrees_of_freedom() was not called");

  params.set("action", "calc_struct_volconstrstiff");

  const bool assmat1 = sysmat1 != nullptr;
  const bool assmat2 = sysmat2 != nullptr;
  const bool assmat3 = sysmat3 != nullptr;
  const bool assvec1 = sysvec1 != nullptr;
  const bool assvec2 = sysvec2 != nullptr;
  const bool assvec3 = sysvec3 != nullptr;
  const bool assvec4 = sysvec4 != nullptr;
  const bool assvec5 = sysvec5 != nullptr;

  // get time-integrator dependent values
  const double theta = params.get("scale_theta", 1.0);
  const double ts_size = params.get("time_step_size", 1.0);

  // global and local ID of this bc in the redundant vectors
  const int offsetID = params.get<int>("OffsetID");
  std::vector<int> gindex(num_dof_);
  gindex[0] = offsetID;
  for (int j = 1; j < num_dof_; j++) gindex[j] = gindex[0] + j;

  bool usetime = true;
  const double tim = params.get("total time", -1.0);
  if (tim < 0.0) usetime = false;

  std::vector<bool> havegid(num_dof_);
  for (int j = 0; j < num_dof_; j++)
  {
    havegid[j] = false;
  }

  // find out whether we will use a time curve and get the factor
  // 0D atrial activation
  double y_at_l_np = 0.0;
  double y_at_r_np = 0.0;
  if (atrium_act_curve_l_ >= 0 && usetime)
    y_at_l_np = Global::Problem::instance()
                    ->function_by_id<Core::Utils::FunctionOfTime>(atrium_act_curve_l_)
                    .evaluate(tim);
  if (atrium_act_curve_r_ >= 0 && usetime)
    y_at_r_np = Global::Problem::instance()
                    ->function_by_id<Core::Utils::FunctionOfTime>(atrium_act_curve_r_)
                    .evaluate(tim);
  // 0D time-varying atrial elastance
  double E_at_l_np = 0.;
  double E_at_r_np = 0.;

  // 0D ventricular activation
  double y_v_l_np = 0.0;
  double y_v_r_np = 0.0;
  if (ventricle_act_curve_l_ >= 0 && usetime)
    y_v_l_np = Global::Problem::instance()
                   ->function_by_id<Core::Utils::FunctionOfTime>(ventricle_act_curve_l_)
                   .evaluate(tim);
  if (ventricle_act_curve_r_ >= 0 && usetime)
    y_v_r_np = Global::Problem::instance()
                   ->function_by_id<Core::Utils::FunctionOfTime>(ventricle_act_curve_r_)
                   .evaluate(tim);
  // 0D time-varying ventricular elastance
  double E_v_l_np = 0.;
  double E_v_r_np = 0.;

  // prescribed atrial elastances
  double E_at_l_prescr_np = 0.0;
  double E_at_r_prescr_np = 0.0;
  if (atrium_prescr_e_curve_l_ >= 0 && usetime)
    E_at_l_prescr_np = Global::Problem::instance()
                           ->function_by_id<Core::Utils::FunctionOfTime>(atrium_prescr_e_curve_l_)
                           .evaluate(tim);
  if (atrium_prescr_e_curve_r_ >= 0 && usetime)
    E_at_r_prescr_np = Global::Problem::instance()
                           ->function_by_id<Core::Utils::FunctionOfTime>(atrium_prescr_e_curve_r_)
                           .evaluate(tim);
  // prescribed ventricular elastances
  double E_v_l_prescr_np = 0.0;
  double E_v_r_prescr_np = 0.0;
  if (ventricle_prescr_e_curve_l_ >= 0 && usetime)
    E_v_l_prescr_np = Global::Problem::instance()
                          ->function_by_id<Core::Utils::FunctionOfTime>(ventricle_prescr_e_curve_l_)
                          .evaluate(tim);
  if (ventricle_prescr_e_curve_r_ >= 0 && usetime)
    E_v_r_prescr_np = Global::Problem::instance()
                          ->function_by_id<Core::Utils::FunctionOfTime>(ventricle_prescr_e_curve_r_)
                          .evaluate(tim);


  switch (atrium_model_)
  {
    case Inpar::Cardiovascular0D::atr_elastance_0d:
    {
      E_at_l_np = (e_at_max_l_ - e_at_min_l_) * y_at_l_np + e_at_min_l_;
      E_at_r_np = (e_at_max_r_ - e_at_min_r_) * y_at_r_np + e_at_min_r_;
    }
    break;
    case Inpar::Cardiovascular0D::atr_structure_3d:
    {
      E_at_l_np = 0.;
      E_at_r_np = 0.;
    }
    break;
    case Inpar::Cardiovascular0D::atr_prescribed:
    {
      E_at_l_np = E_at_l_prescr_np;
      E_at_r_np = E_at_r_prescr_np;
    }
    break;
    default:
      FOUR_C_THROW("Undefined atrium_model!");
      break;
  }

  switch (ventricle_model_)
  {
    case Inpar::Cardiovascular0D::ventr_elastance_0d:
    {
      E_v_l_np = (e_v_max_l_ - e_v_min_l_) * y_v_l_np + e_v_min_l_;
      E_v_r_np = (e_v_max_r_ - e_v_min_r_) * y_v_r_np + e_v_min_r_;
    }
    break;
    case Inpar::Cardiovascular0D::ventr_structure_3d:
    {
      E_v_l_np = 0.;
      E_v_r_np = 0.;
    }
    break;
    case Inpar::Cardiovascular0D::ventr_prescribed:
    {
      E_v_l_np = E_v_l_prescr_np;
      E_v_r_np = E_v_r_prescr_np;
    }
    break;
    default:
      FOUR_C_THROW("Undefined ventricle_model!");
      break;
  }


  // Cardiovascular0D stiffness
  Core::LinAlg::SerialDenseMatrix wkstiff(num_dof_, num_dof_);

  // contributions to total residuals r:
  // r_m = df_m              - f_m
  //     = (df_np - df_n)/dt - theta f_np - (1-theta) f_n
  // here we ONLY evaluate df_np, f_np
  std::vector<double> df_np(num_dof_);
  std::vector<double> f_np(num_dof_);

  // end-point values at t_{n+1}
  double q_vin_l_np = 0.;
  double p_at_l_np = 0.;
  double q_vout_l_np = 0.;
  double p_v_l_np = 0.;
  double p_ar_sys_np = 0.;
  double q_ar_sys_np = 0.;
  double p_arperi_sys_np = 0.;
  double q_arspl_sys_np = 0.;
  double q_arespl_sys_np = 0.;
  double q_armsc_sys_np = 0.;
  double q_arcer_sys_np = 0.;
  double q_arcor_sys_np = 0.;
  double p_venspl_sys_np = 0.;
  double q_venspl_sys_np = 0.;
  double p_venespl_sys_np = 0.;
  double q_venespl_sys_np = 0.;
  double p_venmsc_sys_np = 0.;
  double q_venmsc_sys_np = 0.;
  double p_vencer_sys_np = 0.;
  double q_vencer_sys_np = 0.;
  double p_vencor_sys_np = 0.;
  double q_vencor_sys_np = 0.;
  double p_ven_sys_np = 0.;
  double q_ven_sys_np = 0.;
  double q_vin_r_np = 0.;
  double p_at_r_np = 0.;
  double q_vout_r_np = 0.;
  double p_v_r_np = 0.;
  double p_ar_pul_np = 0.;
  double q_ar_pul_np = 0.;
  double p_cap_pul_np = 0.;
  double q_cap_pul_np = 0.;
  double p_ven_pul_np = 0.;
  double q_ven_pul_np = 0.;
  // 3D ventricular volume at t_{n+1}
  double V_v_l_np = 0.;
  double V_v_r_np = 0.;
  // 3D atrial volume at t_{n+1}
  double V_at_l_np = 0.;
  double V_at_r_np = 0.;

  double R_atvalve_l = 0.;
  double R_arvalve_l = 0.;
  double R_atvalve_r = 0.;
  double R_arvalve_r = 0.;

  if (assvec1 and assvec2 and assvec4 and assvec5)
  {
    // extract values of dof vector at t_{n+1}
    p_at_l_np = (*sysvec4)[0];
    q_vin_l_np = (*sysvec4)[1];
    q_vout_l_np = (*sysvec4)[2];
    p_v_l_np = (*sysvec4)[3];
    p_ar_sys_np = (*sysvec4)[4];
    q_ar_sys_np = (*sysvec4)[5];

    p_arperi_sys_np = (*sysvec4)[6];
    q_arspl_sys_np = (*sysvec4)[7];
    q_arespl_sys_np = (*sysvec4)[8];
    q_armsc_sys_np = (*sysvec4)[9];
    q_arcer_sys_np = (*sysvec4)[10];
    q_arcor_sys_np = (*sysvec4)[11];
    p_venspl_sys_np = (*sysvec4)[12];
    q_venspl_sys_np = (*sysvec4)[13];
    p_venespl_sys_np = (*sysvec4)[14];
    q_venespl_sys_np = (*sysvec4)[15];
    p_venmsc_sys_np = (*sysvec4)[16];
    q_venmsc_sys_np = (*sysvec4)[17];
    p_vencer_sys_np = (*sysvec4)[18];
    q_vencer_sys_np = (*sysvec4)[19];
    p_vencor_sys_np = (*sysvec4)[20];
    q_vencor_sys_np = (*sysvec4)[21];

    p_ven_sys_np = (*sysvec4)[22];
    q_ven_sys_np = (*sysvec4)[23];
    p_at_r_np = (*sysvec4)[24];
    q_vin_r_np = (*sysvec4)[25];
    q_vout_r_np = (*sysvec4)[26];
    p_v_r_np = (*sysvec4)[27];
    p_ar_pul_np = (*sysvec4)[28];
    q_ar_pul_np = (*sysvec4)[29];
    p_cap_pul_np = (*sysvec4)[30];
    q_cap_pul_np = (*sysvec4)[31];
    p_ven_pul_np = (*sysvec4)[32];
    q_ven_pul_np = (*sysvec4)[33];

    // 3D ventricular volume at t_{n+1}
    V_v_l_np = (*sysvec5)[2];
    V_v_r_np = (*sysvec5)[26];
    // 3D atrial volume at t_{n+1}
    V_at_l_np = (*sysvec5)[0];
    V_at_r_np = (*sysvec5)[24];

    switch (atrium_model_)
    {
      case Inpar::Cardiovascular0D::atr_elastance_0d:
      case Inpar::Cardiovascular0D::atr_prescribed:
      {
        df_np[0] = p_at_l_np / E_at_l_np;
        df_np[24] = p_at_r_np / E_at_r_np;
      }
      break;
      case Inpar::Cardiovascular0D::atr_structure_3d:
      {
        df_np[0] = V_at_l_np;
        df_np[24] = V_at_r_np;
      }
      break;
      default:
        FOUR_C_THROW("Undefined atrium_model!");
        break;
    }

    switch (ventricle_model_)
    {
      case Inpar::Cardiovascular0D::ventr_structure_3d:
      {
        df_np[2] = V_v_l_np;
        df_np[26] = V_v_r_np;
      }
      break;
      case Inpar::Cardiovascular0D::ventr_elastance_0d:
      case Inpar::Cardiovascular0D::ventr_prescribed:
      {
        df_np[2] = p_v_l_np / E_v_l_np;
        df_np[26] = p_v_r_np / E_v_r_np;
      }
      break;
      default:
        FOUR_C_THROW("Undefined ventricle_model!");
        break;
    }

    if (p_v_l_np < p_at_l_np) R_atvalve_l = r_atvalve_min_l_;
    if (p_v_l_np >= p_at_l_np) R_atvalve_l = r_atvalve_max_l_;

    if (p_v_l_np < p_ar_sys_np) R_arvalve_l = r_arvalve_max_l_;
    if (p_v_l_np >= p_ar_sys_np) R_arvalve_l = r_arvalve_min_l_;

    if (p_v_r_np < p_at_r_np) R_atvalve_r = r_atvalve_min_r_;
    if (p_v_r_np >= p_at_r_np) R_atvalve_r = r_atvalve_max_r_;

    if (p_v_r_np < p_ar_pul_np) R_arvalve_r = r_arvalve_max_r_;
    if (p_v_r_np >= p_ar_pul_np) R_arvalve_r = r_arvalve_min_r_;

    // df_np[0] see above
    df_np[1] = 0.;
    // df_np[2] see above
    df_np[3] = 0.;
    df_np[4] = c_ar_sys_ * (p_ar_sys_np - z_ar_sys_ * q_vout_l_np);
    df_np[5] = (l_ar_sys_ / r_ar_sys_) * q_ar_sys_np;
    df_np[6] = (c_arspl_sys_ + c_arespl_sys_ + c_armsc_sys_ + c_arcer_sys_ + c_arcor_sys_) *
               p_arperi_sys_np;
    df_np[7] = 0.;
    df_np[8] = 0.;
    df_np[9] = 0.;
    df_np[10] = 0.;
    df_np[11] = 0.;
    df_np[12] = c_venspl_sys_ * p_venspl_sys_np;
    df_np[13] = 0.;
    df_np[14] = c_venespl_sys_ * p_venespl_sys_np;
    df_np[15] = 0.;
    df_np[16] = c_venmsc_sys_ * p_venmsc_sys_np;
    df_np[17] = 0.;
    df_np[18] = c_vencer_sys_ * p_vencer_sys_np;
    df_np[19] = 0.;
    df_np[20] = c_vencor_sys_ * p_vencor_sys_np;
    df_np[21] = 0.;
    df_np[22] = c_ven_sys_ * p_ven_sys_np;
    df_np[23] = (l_ven_sys_ / r_ven_sys_) * q_ven_sys_np;
    // df_np[24] see above
    df_np[25] = 0.;
    // df_np[26] see above
    df_np[27] = 0.;
    df_np[28] = c_ar_pul_ * (p_ar_pul_np - z_ar_pul_ * q_vout_r_np);
    df_np[29] = (l_ar_pul_ / r_ar_pul_) * q_ar_pul_np;
    df_np[30] = c_cap_pul_ * p_cap_pul_np;
    df_np[31] = 0.;
    df_np[32] = c_ven_pul_ * p_ven_pul_np;
    df_np[33] = (l_ven_pul_ / r_ven_pul_) * q_ven_pul_np;

    f_np[0] = -q_ven_pul_np + q_vin_l_np;
    // atrioventricular valve - mitral
    f_np[1] = (p_at_l_np - p_v_l_np) / R_atvalve_l - q_vin_l_np;
    f_np[2] = -q_vin_l_np + q_vout_l_np;
    // semilunar valve - aortic
    f_np[3] = (p_v_l_np - p_ar_sys_np) / R_arvalve_l - q_vout_l_np;
    f_np[4] = -q_vout_l_np + q_ar_sys_np;
    f_np[5] = (p_arperi_sys_np - p_ar_sys_np + z_ar_sys_ * q_vout_l_np) / r_ar_sys_ + q_ar_sys_np;
    f_np[6] = -q_ar_sys_np +
              (q_arspl_sys_np + q_arespl_sys_np + q_armsc_sys_np + q_arcer_sys_np + q_arcor_sys_np);
    f_np[7] = (p_venspl_sys_np - p_arperi_sys_np) / r_arspl_sys_ + q_arspl_sys_np;
    f_np[8] = (p_venespl_sys_np - p_arperi_sys_np) / r_arespl_sys_ + q_arespl_sys_np;
    f_np[9] = (p_venmsc_sys_np - p_arperi_sys_np) / r_armsc_sys_ + q_armsc_sys_np;
    f_np[10] = (p_vencer_sys_np - p_arperi_sys_np) / r_arcer_sys_ + q_arcer_sys_np;
    f_np[11] = (p_vencor_sys_np - p_arperi_sys_np) / r_arcor_sys_ + q_arcor_sys_np;
    f_np[12] = q_venspl_sys_np - q_arspl_sys_np;
    f_np[13] = (p_ven_sys_np - p_venspl_sys_np) / r_venspl_sys_ + q_venspl_sys_np;
    f_np[14] = q_venespl_sys_np - q_arespl_sys_np;
    f_np[15] = (p_ven_sys_np - p_venespl_sys_np) / r_venespl_sys_ + q_venespl_sys_np;
    f_np[16] = q_venmsc_sys_np - q_armsc_sys_np;
    f_np[17] = (p_ven_sys_np - p_venmsc_sys_np) / r_venmsc_sys_ + q_venmsc_sys_np;
    f_np[18] = q_vencer_sys_np - q_arcer_sys_np;
    f_np[19] = (p_ven_sys_np - p_vencer_sys_np) / r_vencer_sys_ + q_vencer_sys_np;
    f_np[20] = q_vencor_sys_np - q_arcor_sys_np;
    f_np[21] = (p_ven_sys_np - p_vencor_sys_np) / r_vencor_sys_ + q_vencor_sys_np;

    f_np[22] = q_ven_sys_np - (q_venspl_sys_np + q_venespl_sys_np + q_venmsc_sys_np +
                                  q_vencer_sys_np + q_vencor_sys_np);

    f_np[23] = (p_at_r_np - p_ven_sys_np) / r_ven_sys_ + q_ven_sys_np;
    f_np[24] = -q_ven_sys_np + q_vin_r_np;
    // atrioventricular valve - tricuspid
    f_np[25] = (p_at_r_np - p_v_r_np) / R_atvalve_r - q_vin_r_np;
    f_np[26] = -q_vin_r_np + q_vout_r_np;
    // semilunar valve - pulmonary
    f_np[27] = (p_v_r_np - p_ar_pul_np) / R_arvalve_r - q_vout_r_np;
    f_np[28] = -q_vout_r_np + q_ar_pul_np;
    f_np[29] = (p_cap_pul_np - p_ar_pul_np + z_ar_pul_ * q_vout_r_np) / r_ar_pul_ + q_ar_pul_np;
    f_np[30] = -q_ar_pul_np + q_cap_pul_np;
    f_np[31] = (p_ven_pul_np - p_cap_pul_np) / r_cap_pul_ + q_cap_pul_np;
    f_np[32] = -q_cap_pul_np + q_ven_pul_np;
    f_np[33] = (p_at_l_np - p_ven_pul_np) / r_ven_pul_ + q_ven_pul_np;

    // insert volumes of all the compartments into vol vector v_np
    if (atrium_model_ == Inpar::Cardiovascular0D::atr_elastance_0d or
        atrium_model_ == Inpar::Cardiovascular0D::atr_prescribed)
    {
      // 0D left atrial volume
      (*sysvec5).get_values()[0] = p_at_l_np / E_at_l_np + v_at_l_u_;
      // 0D right atrial volume
      (*sysvec5).get_values()[24] = p_at_r_np / E_at_r_np + v_at_r_u_;
    }
    if (ventricle_model_ == Inpar::Cardiovascular0D::ventr_elastance_0d or
        ventricle_model_ == Inpar::Cardiovascular0D::ventr_prescribed)
    {
      // 0D left ventricular volume
      (*sysvec5).get_values()[2] = p_v_l_np / E_v_l_np + v_v_l_u_;
      // 0D right ventricular volume
      (*sysvec5).get_values()[26] = p_v_r_np / E_v_r_np + v_v_r_u_;
    }
    // systemic arterial compartment volume
    (*sysvec5).get_values()[4] = c_ar_sys_ * (p_ar_sys_np - z_ar_sys_ * q_vout_l_np) + v_ar_sys_u_;
    // systemic peripheral arterial compartment volume
    (*sysvec5).get_values()[6] =
        (c_arspl_sys_ + c_arespl_sys_ + c_armsc_sys_ + c_arcer_sys_ + c_arcor_sys_) *
            p_arperi_sys_np +
        v_arspl_sys_u_ + v_arespl_sys_u_ + v_armsc_sys_u_ + v_arcer_sys_u_ + v_arcor_sys_u_;


    // systemic venous splanchnic volume
    (*sysvec5).get_values()[12] = c_venspl_sys_ * p_venspl_sys_np + v_venspl_sys_u_;
    // systemic venous extra-splanchnic volume
    (*sysvec5).get_values()[14] = c_venespl_sys_ * p_venespl_sys_np + v_venespl_sys_u_;
    // systemic venous musclular volume
    (*sysvec5).get_values()[16] = c_venmsc_sys_ * p_venmsc_sys_np + v_venmsc_sys_u_;
    // systemic venous cerebral volume
    (*sysvec5).get_values()[18] = c_vencer_sys_ * p_vencer_sys_np + v_vencer_sys_u_;
    // systemic venous coronary volume
    (*sysvec5).get_values()[20] = c_vencor_sys_ * p_vencor_sys_np + v_vencor_sys_u_;

    // systemic venous compartment volume
    (*sysvec5).get_values()[22] = c_ven_sys_ * p_ven_sys_np + v_ven_sys_u_;
    // pulmonary arterial compartment volume
    (*sysvec5).get_values()[28] = c_ar_pul_ * (p_ar_pul_np - z_ar_pul_ * q_vout_r_np) + v_ar_pul_u_;
    // pulmonary capillary volume
    (*sysvec5).get_values()[30] = c_cap_pul_ * p_cap_pul_np + v_cap_pul_u_;
    // pulmonary venous compartment volume
    (*sysvec5).get_values()[32] = c_ven_pul_ * p_ven_pul_np + v_ven_pul_u_;

    // call sub evaluate method for respiratory model
    // after all vascular compartment volumes have been set - since these enter the 0D respiratory
    // residual!!!
    switch (respiratory_model_)
    {
      case Inpar::Cardiovascular0D::resp_none:
        break;
      case Inpar::Cardiovascular0D::resp_standard:
        evaluate_respiratory(params, df_np, f_np, wkstiff, sysvec4, sysvec5, false);
        break;
    }
  }


  // assemble of Cardiovascular0D stiffness matrix, scale with time-integrator dependent value
  if (assmat1)
  {
    // atrium - left and right
    switch (atrium_model_)
    {
      case Inpar::Cardiovascular0D::atr_elastance_0d:
      case Inpar::Cardiovascular0D::atr_prescribed:
        wkstiff(0, 0) = 1. / (E_at_l_np * ts_size);
        wkstiff(24, 24) = 1. / (E_at_r_np * ts_size);
        break;
      case Inpar::Cardiovascular0D::atr_structure_3d:
        wkstiff(0, 0) = 0.;
        wkstiff(24, 24) = 0.;
        break;
      default:
        FOUR_C_THROW("Undefined atrium_model!");
        break;
    }

    // ventricle - left and right
    switch (ventricle_model_)
    {
      case Inpar::Cardiovascular0D::ventr_structure_3d:
        wkstiff(2, 3) = 0.;
        wkstiff(26, 27) = 0.;
        break;
      case Inpar::Cardiovascular0D::ventr_elastance_0d:
      case Inpar::Cardiovascular0D::ventr_prescribed:
        wkstiff(2, 3) = 1. / (E_v_l_np * ts_size);
        wkstiff(26, 27) = 1. / (E_v_r_np * ts_size);
        break;
      default:
        FOUR_C_THROW("Undefined ventricle_model!");
        break;
    }

    // atrium - left
    // wkstiff(0,0) see above
    wkstiff(0, 1) = theta;
    wkstiff(0, 33) = -theta;

    // atrioventricular valve - mitral
    wkstiff(1, 0) = theta / R_atvalve_l;
    wkstiff(1, 1) = -theta;
    wkstiff(1, 3) = -theta / R_atvalve_l;

    // ventricular mass balance - left
    wkstiff(2, 1) = -theta;
    wkstiff(2, 2) = theta;
    // wkstiff(2,3) see above

    // semilunar valve - aortic
    wkstiff(3, 2) = -theta;
    wkstiff(3, 3) = theta / R_arvalve_l;
    wkstiff(3, 4) = -theta / R_arvalve_l;

    // arterial mass balance - systemic
    wkstiff(4, 2) = -theta - c_ar_sys_ * z_ar_sys_ / ts_size;
    wkstiff(4, 4) = c_ar_sys_ / ts_size;
    wkstiff(4, 5) = theta;

    // arterial linear momentum balance - systemic
    wkstiff(5, 2) = z_ar_sys_ * theta / r_ar_sys_;
    wkstiff(5, 4) = -theta / r_ar_sys_;
    wkstiff(5, 5) = l_ar_sys_ / (r_ar_sys_ * ts_size) + theta;
    wkstiff(5, 6) = theta / r_ar_sys_;

    wkstiff(6, 5) = -theta;
    wkstiff(6, 6) =
        (c_arspl_sys_ + c_arespl_sys_ + c_armsc_sys_ + c_arcer_sys_ + c_arcor_sys_) / ts_size;
    wkstiff(6, 7) = theta;
    wkstiff(6, 8) = theta;
    wkstiff(6, 9) = theta;
    wkstiff(6, 10) = theta;
    wkstiff(6, 11) = theta;


    wkstiff(7, 6) = -theta / r_arspl_sys_;
    wkstiff(7, 7) = theta;
    wkstiff(7, 12) = theta / r_arspl_sys_;

    wkstiff(8, 6) = -theta / r_arespl_sys_;
    wkstiff(8, 8) = theta;
    wkstiff(8, 14) = theta / r_arespl_sys_;

    wkstiff(9, 6) = -theta / r_armsc_sys_;
    wkstiff(9, 9) = theta;
    wkstiff(9, 16) = theta / r_armsc_sys_;

    wkstiff(10, 6) = -theta / r_arcer_sys_;
    wkstiff(10, 10) = theta;
    wkstiff(10, 18) = theta / r_arcer_sys_;

    wkstiff(11, 6) = -theta / r_arcor_sys_;
    wkstiff(11, 11) = theta;
    wkstiff(11, 20) = theta / r_arcor_sys_;

    wkstiff(12, 7) = -theta;
    wkstiff(12, 12) = c_venspl_sys_ / ts_size;
    wkstiff(12, 13) = theta;

    wkstiff(13, 12) = -theta / r_venspl_sys_;
    wkstiff(13, 13) = theta;
    wkstiff(13, 22) = theta / r_venspl_sys_;

    wkstiff(14, 8) = -theta;
    wkstiff(14, 14) = c_venespl_sys_ / ts_size;
    wkstiff(14, 15) = theta;

    wkstiff(15, 14) = -theta / r_venespl_sys_;
    wkstiff(15, 15) = theta;
    wkstiff(15, 22) = theta / r_venespl_sys_;

    wkstiff(16, 9) = -theta;
    wkstiff(16, 16) = c_venmsc_sys_ / ts_size;
    wkstiff(16, 17) = theta;

    wkstiff(17, 16) = -theta / r_venmsc_sys_;
    wkstiff(17, 17) = theta;
    wkstiff(17, 22) = theta / r_venmsc_sys_;

    wkstiff(18, 10) = -theta;
    wkstiff(18, 18) = c_vencer_sys_ / ts_size;
    wkstiff(18, 19) = theta;

    wkstiff(19, 18) = -theta / r_vencer_sys_;
    wkstiff(19, 19) = theta;
    wkstiff(19, 22) = theta / r_vencer_sys_;

    wkstiff(20, 11) = -theta;
    wkstiff(20, 20) = c_vencor_sys_ / ts_size;
    wkstiff(20, 21) = theta;

    wkstiff(21, 20) = -theta / r_vencor_sys_;
    wkstiff(21, 21) = theta;
    wkstiff(21, 22) = theta / r_vencor_sys_;

    wkstiff(22, 13) = -theta;
    wkstiff(22, 15) = -theta;
    wkstiff(22, 17) = -theta;
    wkstiff(22, 19) = -theta;
    wkstiff(22, 21) = -theta;
    wkstiff(22, 22) = c_ven_sys_ / ts_size;
    wkstiff(22, 23) = theta;

    wkstiff(23, 22) = -theta / r_ven_sys_;
    wkstiff(23, 23) = l_ven_sys_ / (r_ven_sys_ * ts_size) + theta;
    wkstiff(23, 24) = theta / r_ven_sys_;

    // atrium - right
    wkstiff(24, 23) = -theta;
    // wkstiff(24,24) see above
    wkstiff(24, 25) = theta;

    // atrioventricular valve - tricuspid
    wkstiff(25, 24) = theta / R_atvalve_r;
    wkstiff(25, 25) = -theta;
    wkstiff(25, 27) = -theta / R_atvalve_r;

    // ventricular mass balance - right
    wkstiff(26, 25) = -theta;
    wkstiff(26, 26) = theta;
    // wkstiff(26,27) see above

    // semilunar valve - pulmonary
    wkstiff(27, 26) = -theta;
    wkstiff(27, 27) = theta / R_arvalve_r;
    wkstiff(27, 28) = -theta / R_arvalve_r;

    // arterial mass balance - pulmonary
    wkstiff(28, 26) = -theta - c_ar_pul_ * z_ar_pul_ / ts_size;
    wkstiff(28, 28) = c_ar_pul_ / ts_size;
    wkstiff(28, 29) = theta;

    // arterial linear momentum balance - pulmonary
    wkstiff(29, 26) = z_ar_pul_ * theta / r_ar_pul_;
    wkstiff(29, 28) = -theta / r_ar_pul_;
    wkstiff(29, 29) = l_ar_pul_ / (r_ar_pul_ * ts_size) + theta;
    wkstiff(29, 30) = theta / r_ar_pul_;

    wkstiff(30, 29) = -theta;
    wkstiff(30, 30) = c_cap_pul_ / ts_size;
    wkstiff(30, 31) = theta;

    wkstiff(31, 30) = -theta / r_cap_pul_;
    wkstiff(31, 31) = theta;
    wkstiff(31, 32) = theta / r_cap_pul_;

    // venous mass balance - pulmonary
    wkstiff(32, 31) = -theta;
    wkstiff(32, 32) = c_ven_pul_ / ts_size;
    wkstiff(32, 33) = theta;

    // venous linear momentum balance - pulmonary
    wkstiff(33, 0) = theta / r_ven_pul_;
    wkstiff(33, 32) = -theta / r_ven_pul_;
    wkstiff(33, 33) = l_ven_pul_ / (r_ven_pul_ * ts_size) + theta;


    // call sub evaluate method for respiratory model
    switch (respiratory_model_)
    {
      case Inpar::Cardiovascular0D::resp_none:
        break;
      case Inpar::Cardiovascular0D::resp_standard:
        evaluate_respiratory(params, df_np, f_np, wkstiff, sysvec4, sysvec5, true);
        break;
    }

    sysmat1->un_complete();

    // assemble into cardiovascular0d system matrix - wkstiff contribution
    for (int j = 0; j < num_dof_; j++)
    {
      for (int k = 0; k < num_dof_; k++)
      {
        havegid[k] = sysmat1->row_map().my_gid(gindex[k]);
        if (havegid[k])
        {
          sysmat1->assemble(wkstiff(k, j), gindex[k], gindex[j]);
        }
      }
    }
  }
  // rhs part df_np
  if (assvec1)
  {
    for (int j = 0; j < num_dof_; j++)
    {
      sysvec1->sum_into_global_values(1, &df_np[j], &gindex[j]);
    }
  }
  // rhs part f_np
  if (assvec2)
  {
    for (int j = 0; j < num_dof_; j++)
    {
      sysvec2->sum_into_global_values(1, &f_np[j], &gindex[j]);
    }
  }


  //----------------------------------------------------------------------
  // loop through conditions and evaluate them if they match the criterion
  //----------------------------------------------------------------------
  for (unsigned int i = 0; i < cardiovascular0dcond_.size(); ++i)
  {
    const Core::Conditions::Condition& cond = *(cardiovascular0dcond_[i]);

    // elements might need condition
    params.set<const Core::Conditions::Condition*>("condition", &cond);

    const std::string conditiontype =
        cardiovascular0dcond_[i]->parameters().get<std::string>("TYPE");

    // define element matrices and vectors
    Core::LinAlg::SerialDenseMatrix elematrix1;
    Core::LinAlg::SerialDenseMatrix elematrix2;
    Core::LinAlg::SerialDenseVector elevector1;
    Core::LinAlg::SerialDenseVector elevector2;
    Core::LinAlg::SerialDenseVector elevector2a;
    Core::LinAlg::SerialDenseVector elevector2b;
    Core::LinAlg::SerialDenseVector elevector3;

    const std::map<int, std::shared_ptr<Core::Elements::Element>>& geom = cond.geometry();
    // if (geom.empty()) FOUR_C_THROW("evaluation of condition with empty geometry");
    // no check for empty geometry here since in parallel computations
    // can exist processors which do not own a portion of the elements belonging
    // to the condition geometry
    for (const auto& [id, ele] : geom)
    {
      // get element location vector and ownerships
      std::vector<int> lm;
      std::vector<int> lmowner;
      std::vector<int> lmstride;
      ele->location_vector(*actdisc_, lm, lmowner, lmstride);

      // get dimension of element matrices and vectors
      // Reshape element matrices and vectors and init to zero
      const int eledim = (int)lm.size();

      elematrix2.shape(eledim, eledim);
      elevector2.size(eledim);
      elevector2a.size(eledim);
      elevector2b.size(eledim);
      elevector3.size(1);

      for (int k = 0; k < eledim; k++)
      {
        elevector2a[k] = elevector2[k];
        elevector2b[k] = elevector2[k];
      }

      // call the element specific evaluate method
      int err = ele->evaluate(
          params, *actdisc_, lm, elematrix1, elematrix2, elevector1, elevector2, elevector3);
      if (err) FOUR_C_THROW("error while evaluating elements");


      // assembly
      int eid = ele->id();

      if (assmat2 and conditiontype != "dummy")
      {
        // assemble the offdiagonal stiffness block (1,0 block) arising from dR_cardvasc0d/dd
        // -> this matrix is later on transposed when building the whole block matrix
        std::vector<int> colvec(1);
        std::vector<int> colvec_a(1);
        std::vector<int> colvec_b(1);

        // consistent linearization: include further derivatives w.r.t. to structural displacement
        // in case of respiratory model, since ventricular and atrial volumes appear in the
        // transport residual expressions
        switch (respiratory_model_)
        {
          case Inpar::Cardiovascular0D::resp_none:
          {
            if (conditiontype == "ventricle_left") colvec[0] = gindex[2];
            if (conditiontype == "ventricle_right") colvec[0] = gindex[26];
            if (conditiontype == "atrium_left") colvec[0] = gindex[0];
            if (conditiontype == "atrium_right") colvec[0] = gindex[24];
            elevector2.scale(-1. / ts_size);
            sysmat2->assemble(eid, lmstride, elevector2, lm, lmowner, colvec);
          }
          break;
          case Inpar::Cardiovascular0D::resp_standard:
          {
            if (conditiontype == "ventricle_left")
            {
              colvec[0] = gindex[2];
              elevector2.scale(-1. / ts_size);
              sysmat2->assemble(eid, lmstride, elevector2, lm, lmowner, colvec);
              colvec_a[0] = gindex[56];
              elevector2a.scale(-f_np[56] / V_v_l_np);
              sysmat2->assemble(eid, lmstride, elevector2a, lm, lmowner, colvec_a);
              colvec_b[0] = gindex[57];
              elevector2b.scale(-f_np[57] / V_v_l_np);
              sysmat2->assemble(eid, lmstride, elevector2b, lm, lmowner, colvec_b);
            }
            if (conditiontype == "ventricle_right")
            {
              colvec[0] = gindex[26];
              elevector2.scale(-1. / ts_size);
              sysmat2->assemble(eid, lmstride, elevector2, lm, lmowner, colvec);
              colvec_a[0] = gindex[46];
              elevector2a.scale(-f_np[46] / V_v_r_np);
              sysmat2->assemble(eid, lmstride, elevector2a, lm, lmowner, colvec_a);
              colvec_b[0] = gindex[47];
              elevector2b.scale(-f_np[47] / V_v_r_np);
              sysmat2->assemble(eid, lmstride, elevector2b, lm, lmowner, colvec_b);
            }
            if (conditiontype == "atrium_left")
            {
              colvec[0] = gindex[0];
              elevector2.scale(-1. / ts_size);
              sysmat2->assemble(eid, lmstride, elevector2, lm, lmowner, colvec);
              colvec_a[0] = gindex[54];
              elevector2a.scale(-f_np[54] / V_at_l_np);
              sysmat2->assemble(eid, lmstride, elevector2a, lm, lmowner, colvec_a);
              colvec_b[0] = gindex[55];
              elevector2b.scale(-f_np[55] / V_at_l_np);
              sysmat2->assemble(eid, lmstride, elevector2b, lm, lmowner, colvec_b);
            }
            if (conditiontype == "atrium_right")
            {
              colvec[0] = gindex[24];
              elevector2.scale(-1. / ts_size);
              sysmat2->assemble(eid, lmstride, elevector2, lm, lmowner, colvec);
              colvec_a[0] = gindex[44];
              elevector2a.scale(-f_np[44] / V_at_r_np);
              sysmat2->assemble(eid, lmstride, elevector2a, lm, lmowner, colvec_a);
              colvec_b[0] = gindex[45];
              elevector2b.scale(-f_np[45] / V_at_r_np);
              sysmat2->assemble(eid, lmstride, elevector2b, lm, lmowner, colvec_b);
            }
          }
          break;
        }
      }
      if (assvec3 and conditiontype != "dummy")
      {
        // assemble the current volume of the enclosed surface of the cardiovascular0d condition
        std::vector<int> cardiovascular0dlm;
        std::vector<int> cardiovascular0downer;

        if (conditiontype == "ventricle_left") cardiovascular0dlm.push_back(gindex[2]);
        if (conditiontype == "ventricle_right") cardiovascular0dlm.push_back(gindex[26]);
        if (conditiontype == "atrium_left") cardiovascular0dlm.push_back(gindex[0]);
        if (conditiontype == "atrium_right") cardiovascular0dlm.push_back(gindex[24]);
        cardiovascular0downer.push_back(ele->owner());
        Core::LinAlg::assemble(*sysvec3, elevector3, cardiovascular0dlm, cardiovascular0downer);
      }
    }
  }

  if (assmat3)
  {
    // offdiagonal stiffness block (0,1 block)
    evaluate_d_struct_dp(params, *sysmat3);
  }
}  // end of evaluate_condition



void Utils::CardiovascularRespiratory0DSysPulPeriphCirculation::evaluate_respiratory(
    Teuchos::ParameterList& params, std::vector<double>& df_np, std::vector<double>& f_np,
    Core::LinAlg::SerialDenseMatrix& wkstiff, std::shared_ptr<Core::LinAlg::Vector<double>> dofvec,
    std::shared_ptr<Core::LinAlg::Vector<double>> volvec, bool evalstiff)
{
  // get time-integrator dependent values
  const double theta = params.get("scale_theta", 1.0);
  const double ts_size = params.get("time_step_size", 1.0);

  bool usetime = true;
  const double tim = params.get("total time", -1.0);
  if (tim < 0.0) usetime = false;

  // find out whether we will use a time curve and get the factor
  double U_t = 0.0;
  if (u_t_curve_ >= 0 && usetime)
    U_t = Global::Problem::instance()
              ->function_by_id<Core::Utils::FunctionOfTime>(u_t_curve_)
              .evaluate(tim);

  // extract values of dof vector at t_{n+1}
  const double p_at_l_np = (*dofvec)[0];
  const double q_vin_l_np = (*dofvec)[1];
  const double q_vout_l_np = (*dofvec)[2];
  const double p_v_l_np = (*dofvec)[3];
  const double p_ar_sys_np = (*dofvec)[4];
  //  const double q_ar_sys_np = (*dofvec)[5];

  const double p_arperi_sys_np = (*dofvec)[6];
  const double q_arspl_sys_np = (*dofvec)[7];
  const double q_arespl_sys_np = (*dofvec)[8];
  const double q_armsc_sys_np = (*dofvec)[9];
  const double q_arcer_sys_np = (*dofvec)[10];
  const double q_arcor_sys_np = (*dofvec)[11];
  const double p_venspl_sys_np = (*dofvec)[12];
  const double q_venspl_sys_np = (*dofvec)[13];
  const double p_venespl_sys_np = (*dofvec)[14];
  const double q_venespl_sys_np = (*dofvec)[15];
  const double p_venmsc_sys_np = (*dofvec)[16];
  const double q_venmsc_sys_np = (*dofvec)[17];
  const double p_vencer_sys_np = (*dofvec)[18];
  const double q_vencer_sys_np = (*dofvec)[19];
  const double p_vencor_sys_np = (*dofvec)[20];
  const double q_vencor_sys_np = (*dofvec)[21];

  const double p_ven_sys_np = (*dofvec)[22];
  const double q_ven_sys_np = (*dofvec)[23];
  const double p_at_r_np = (*dofvec)[24];
  const double q_vin_r_np = (*dofvec)[25];
  const double q_vout_r_np = (*dofvec)[26];
  const double p_v_r_np = (*dofvec)[27];
  const double p_ar_pul_np = (*dofvec)[28];
  const double q_ar_pul_np = (*dofvec)[29];
  const double p_cap_pul_np = (*dofvec)[30];
  const double q_cap_pul_np = (*dofvec)[31];
  const double p_ven_pul_np = (*dofvec)[32];
  const double q_ven_pul_np = (*dofvec)[33];

  const double V_alv_np = (*dofvec)[34];
  const double q_alv_np = (*dofvec)[35];
  const double p_alv_np = (*dofvec)[36];
  const double fCO2_alv_np = (*dofvec)[37];
  const double fO2_alv_np = (*dofvec)[38];

  const double q_arspl_sys_in_np = (*dofvec)[39];
  const double q_arespl_sys_in_np = (*dofvec)[40];
  const double q_armsc_sys_in_np = (*dofvec)[41];
  const double q_arcer_sys_in_np = (*dofvec)[42];
  const double q_arcor_sys_in_np = (*dofvec)[43];

  const double ppCO2_at_r_np = (*dofvec)[44];
  const double ppO2_at_r_np = (*dofvec)[45];
  const double ppCO2_v_r_np = (*dofvec)[46];
  const double ppO2_v_r_np = (*dofvec)[47];
  const double ppCO2_ar_pul_np = (*dofvec)[48];
  const double ppO2_ar_pul_np = (*dofvec)[49];
  // gas partial pressures at pulmonary capillaries
  const double ppCO2_cap_pul_np = (*dofvec)[50];
  const double ppO2_cap_pul_np = (*dofvec)[51];

  const double ppCO2_ven_pul_np = (*dofvec)[52];
  const double ppO2_ven_pul_np = (*dofvec)[53];
  const double ppCO2_at_l_np = (*dofvec)[54];
  const double ppO2_at_l_np = (*dofvec)[55];
  const double ppCO2_v_l_np = (*dofvec)[56];
  const double ppO2_v_l_np = (*dofvec)[57];
  const double ppCO2_ar_sys_np = (*dofvec)[58];
  const double ppO2_ar_sys_np = (*dofvec)[59];

  // gas partial pressures at systemic capillaries
  const double ppCO2_arspl_sys_np = (*dofvec)[60];
  const double ppO2_arspl_sys_np = (*dofvec)[61];
  const double ppCO2_arespl_sys_np = (*dofvec)[62];
  const double ppO2_arespl_sys_np = (*dofvec)[63];
  const double ppCO2_armsc_sys_np = (*dofvec)[64];
  const double ppO2_armsc_sys_np = (*dofvec)[65];
  const double ppCO2_arcer_sys_np = (*dofvec)[66];
  const double ppO2_arcer_sys_np = (*dofvec)[67];
  const double ppCO2_arcor_sys_np = (*dofvec)[68];
  const double ppO2_arcor_sys_np = (*dofvec)[69];

  const double ppCO2_venspl_sys_np = (*dofvec)[70];
  const double ppO2_venspl_sys_np = (*dofvec)[71];
  const double ppCO2_venespl_sys_np = (*dofvec)[72];
  const double ppO2_venespl_sys_np = (*dofvec)[73];
  const double ppCO2_venmsc_sys_np = (*dofvec)[74];
  const double ppO2_venmsc_sys_np = (*dofvec)[75];
  const double ppCO2_vencer_sys_np = (*dofvec)[76];
  const double ppO2_vencer_sys_np = (*dofvec)[77];
  const double ppCO2_vencor_sys_np = (*dofvec)[78];
  const double ppO2_vencor_sys_np = (*dofvec)[79];
  const double ppCO2_ven_sys_np = (*dofvec)[80];
  const double ppO2_ven_sys_np = (*dofvec)[81];


  // volumes at t_{n+1} - for transport and dissociation models
  const double V_at_l_np = (*volvec)[0];
  const double V_v_l_np = (*volvec)[2];
  const double V_at_r_np = (*volvec)[24];
  const double V_v_r_np = (*volvec)[26];
  // systemic arterial compartment volume
  const double V_ar_sys_np = c_ar_sys_ * (p_ar_sys_np - z_ar_sys_ * q_vout_l_np) + v_ar_sys_u_;
  // systemic peripheral arterial compartment volume
  const double V_arspl_sys_np = c_arspl_sys_ * p_arperi_sys_np + v_arspl_sys_u_;
  const double V_arespl_sys_np = c_arespl_sys_ * p_arperi_sys_np + v_arespl_sys_u_;
  const double V_armsc_sys_np = c_armsc_sys_ * p_arperi_sys_np + v_armsc_sys_u_;
  const double V_arcer_sys_np = c_arcer_sys_ * p_arperi_sys_np + v_arcer_sys_u_;
  const double V_arcor_sys_np = c_arcor_sys_ * p_arperi_sys_np + v_arcor_sys_u_;
  // systemic venous splanchnic volume
  const double V_venspl_sys_np = c_venspl_sys_ * p_venspl_sys_np + v_venspl_sys_u_;
  // systemic venous extra-splanchnic volume
  const double V_venespl_sys_np = c_venespl_sys_ * p_venespl_sys_np + v_venespl_sys_u_;
  // systemic venous musclular volume
  const double V_venmsc_sys_np = c_venmsc_sys_ * p_venmsc_sys_np + v_venmsc_sys_u_;
  // systemic venous cerebral volume
  const double V_vencer_sys_np = c_vencer_sys_ * p_vencer_sys_np + v_vencer_sys_u_;
  // systemic venous coronary volume
  const double V_vencor_sys_np = c_vencor_sys_ * p_vencor_sys_np + v_vencor_sys_u_;
  // systemic venous compartment volume
  const double V_ven_sys_np = c_ven_sys_ * p_ven_sys_np + v_ven_sys_u_;
  // pulmonary arterial compartment volume
  const double V_ar_pul_np = c_ar_pul_ * (p_ar_pul_np - z_ar_pul_ * q_vout_r_np) + v_ar_pul_u_;
  // pulmonary capillary volume
  const double V_cap_pul_np = c_cap_pul_ * p_cap_pul_np + v_cap_pul_u_;
  // pulmonary venous compartment volume
  const double V_ven_pul_np = c_ven_pul_ * p_ven_pul_np + v_ven_pul_u_;


  // alveolar volume
  (*volvec).get_values()[34] = V_alv_np;

  // we misuse the vol vector to carry information about the O2 saturation S_O2 of the respective
  // compartment in order to avoid introducing another vector for this purpose the vol vector
  // has plenty of zero entries after LID 34, and it is time-integrated and post-processed to
  // t_{n+\theta} inside the manager

  // pulmonary arterial O2 saturation
  (*volvec).get_values()[49] = s_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np);
  // systemic arterial O2 saturation
  (*volvec).get_values()[59] = s_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np);

  // contributions to residual
  // 0D lung
  df_np[34] = V_alv_np;
  df_np[35] = l_alv_ * q_alv_np;
  df_np[36] = p_alv_np;
  f_np[34] = -q_alv_np;
  f_np[35] = r_alv_ * q_alv_np + e_alv_ * (V_alv_np - v_lung_u_) - p_alv_np + U_t;
  f_np[36] = -(1. / V_alv_np) *
             (u_m_ * ((u_m_ - p_alv_np) / r_alv_ +
                         v_m_gas_ * kappa_c_o2_ *
                             (ppCO2_cap_pul_np - fCO2_alv_np * (p_alv_np - p_vap_water_37_)) +
                         v_m_gas_ * kappa_o2_ *
                             (ppO2_cap_pul_np - fO2_alv_np * (p_alv_np - p_vap_water_37_))) -
                 p_alv_np * q_alv_np);

  double fCO2_insp = 0.;
  double fO2_insp = 0.;

  if (v_lung_tidal_ >= v_lung_dead_)
    fCO2_insp =
        (fCO2_alv_np * v_lung_dead_ + f_c_o2_ext_ * (v_lung_tidal_ - v_lung_dead_)) / v_lung_tidal_;
  if (v_lung_tidal_ < v_lung_dead_) fCO2_insp = fCO2_alv_np;

  if (v_lung_tidal_ >= v_lung_dead_)
    fO2_insp =
        (fO2_alv_np * v_lung_dead_ + f_o2_ext_ * (v_lung_tidal_ - v_lung_dead_)) / v_lung_tidal_;
  if (v_lung_tidal_ < v_lung_dead_) fO2_insp = fO2_alv_np;

  double q_insp = 0.;

  if ((u_m_ - p_alv_np) / r_alv_ > 0.) q_insp = (u_m_ - p_alv_np) / r_alv_;
  if ((u_m_ - p_alv_np) / r_alv_ <= 0.) q_insp = 0.;

  df_np[37] = fCO2_alv_np;
  df_np[38] = fO2_alv_np;
  f_np[37] =
      -(1. / V_alv_np) *
      (v_m_gas_ * kappa_c_o2_ * (ppCO2_cap_pul_np - fCO2_alv_np * p_alv_np) +
          (fCO2_insp - fCO2_alv_np) * q_insp -
          fCO2_alv_np * (v_m_gas_ * kappa_o2_ *
                                (ppO2_cap_pul_np - fO2_alv_np * (p_alv_np - p_vap_water_37_)) +
                            v_m_gas_ * kappa_c_o2_ *
                                (ppCO2_cap_pul_np - fCO2_alv_np * (p_alv_np - p_vap_water_37_))));
  f_np[38] =
      -(1. / V_alv_np) *
      (v_m_gas_ * kappa_o2_ * (ppO2_cap_pul_np - fO2_alv_np * p_alv_np) +
          (fO2_insp - fO2_alv_np) * q_insp -
          fO2_alv_np * (v_m_gas_ * kappa_c_o2_ *
                               (ppCO2_cap_pul_np - fCO2_alv_np * (p_alv_np - p_vap_water_37_)) +
                           v_m_gas_ * kappa_o2_ *
                               (ppO2_cap_pul_np - fO2_alv_np * (p_alv_np - p_vap_water_37_))));

  df_np[39] = c_arspl_sys_ * p_arperi_sys_np;
  df_np[40] = c_arespl_sys_ * p_arperi_sys_np;
  df_np[41] = c_armsc_sys_ * p_arperi_sys_np;
  df_np[42] = c_arcer_sys_ * p_arperi_sys_np;
  df_np[43] = c_arcor_sys_ * p_arperi_sys_np;
  f_np[39] = q_arspl_sys_np - q_arspl_sys_in_np;
  f_np[40] = q_arespl_sys_np - q_arespl_sys_in_np;
  f_np[41] = q_armsc_sys_np - q_armsc_sys_in_np;
  f_np[42] = q_arcer_sys_np - q_arcer_sys_in_np;
  f_np[43] = q_arcor_sys_np - q_arcor_sys_in_np;


  // gas transport in cardiovascular system
  df_np[44] = ppCO2_at_r_np;
  df_np[45] = ppO2_at_r_np;
  df_np[46] = ppCO2_v_r_np;
  df_np[47] = ppO2_v_r_np;
  df_np[48] = ppCO2_ar_pul_np;
  df_np[49] = ppO2_ar_pul_np;

  // gas partial pressures at systemic capillaries
  df_np[50] = ppCO2_cap_pul_np;
  df_np[51] = ppO2_cap_pul_np;

  df_np[52] = ppCO2_ven_pul_np;
  df_np[53] = ppO2_ven_pul_np;
  df_np[54] = ppCO2_at_l_np;
  df_np[55] = ppO2_at_l_np;
  df_np[56] = ppCO2_v_l_np;
  df_np[57] = ppO2_v_l_np;
  df_np[58] = ppCO2_ar_sys_np;
  df_np[59] = ppO2_ar_sys_np;

  // gas partial pressures at systemic capillaries
  // arterioles
  df_np[60] = ppCO2_arspl_sys_np;
  df_np[61] = ppO2_arspl_sys_np;
  df_np[62] = ppCO2_arespl_sys_np;
  df_np[63] = ppO2_arespl_sys_np;
  df_np[64] = ppCO2_armsc_sys_np;
  df_np[65] = ppO2_armsc_sys_np;
  df_np[66] = ppCO2_arcer_sys_np;
  df_np[67] = ppO2_arcer_sys_np;
  df_np[68] = ppCO2_arcor_sys_np;
  df_np[69] = ppO2_arcor_sys_np;
  // venules
  df_np[70] = ppCO2_venspl_sys_np;
  df_np[71] = ppO2_venspl_sys_np;
  df_np[72] = ppCO2_venespl_sys_np;
  df_np[73] = ppO2_venespl_sys_np;
  df_np[74] = ppCO2_venmsc_sys_np;
  df_np[75] = ppO2_venmsc_sys_np;
  df_np[76] = ppCO2_vencer_sys_np;
  df_np[77] = ppO2_vencer_sys_np;
  df_np[78] = ppCO2_vencor_sys_np;
  df_np[79] = ppO2_vencor_sys_np;
  df_np[80] = ppCO2_ven_sys_np;
  df_np[81] = ppO2_ven_sys_np;

  // right atrium CO2
  f_np[44] = (1. / V_at_r_np) *
             pow((dcb_c_o2_dpp_c_o2(ppCO2_at_r_np, ppO2_at_r_np) *
                         dcb_o2_dpp_o2(ppCO2_at_r_np, ppO2_at_r_np) -
                     dcb_o2_dpp_c_o2(ppCO2_at_r_np, ppO2_at_r_np) *
                         dcb_c_o2_dpp_o2(ppCO2_at_r_np, ppO2_at_r_np)),
                 -1.) *
             (dcb_o2_dpp_o2(ppCO2_at_r_np, ppO2_at_r_np) *
                     (q_ven_sys_np * (cb_c_o2(ppCO2_at_r_np, ppO2_at_r_np) -
                                         cb_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np))) -
                 dcb_c_o2_dpp_o2(ppCO2_at_r_np, ppO2_at_r_np) *
                     (q_ven_sys_np * (cb_o2(ppCO2_at_r_np, ppO2_at_r_np) -
                                         cb_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np))));
  // right atrium O2
  f_np[45] = (1. / V_at_r_np) *
             pow((dcb_c_o2_dpp_c_o2(ppCO2_at_r_np, ppO2_at_r_np) *
                         dcb_o2_dpp_o2(ppCO2_at_r_np, ppO2_at_r_np) -
                     dcb_o2_dpp_c_o2(ppCO2_at_r_np, ppO2_at_r_np) *
                         dcb_c_o2_dpp_o2(ppCO2_at_r_np, ppO2_at_r_np)),
                 -1.) *
             (dcb_c_o2_dpp_c_o2(ppCO2_at_r_np, ppO2_at_r_np) *
                     (q_ven_sys_np * (cb_o2(ppCO2_at_r_np, ppO2_at_r_np) -
                                         cb_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np))) -
                 dcb_o2_dpp_c_o2(ppCO2_at_r_np, ppO2_at_r_np) *
                     (q_ven_sys_np * (cb_c_o2(ppCO2_at_r_np, ppO2_at_r_np) -
                                         cb_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np))));

  // right ventricle CO2
  f_np[46] =
      (1. / V_v_r_np) *
      pow((dcb_c_o2_dpp_c_o2(ppCO2_v_r_np, ppO2_v_r_np) * dcb_o2_dpp_o2(ppCO2_v_r_np, ppO2_v_r_np) -
              dcb_o2_dpp_c_o2(ppCO2_v_r_np, ppO2_v_r_np) *
                  dcb_c_o2_dpp_o2(ppCO2_v_r_np, ppO2_v_r_np)),
          -1.) *
      (dcb_o2_dpp_o2(ppCO2_v_r_np, ppO2_v_r_np) *
              (q_vin_r_np *
                  (cb_c_o2(ppCO2_v_r_np, ppO2_v_r_np) - cb_c_o2(ppCO2_at_r_np, ppO2_at_r_np))) -
          dcb_c_o2_dpp_o2(ppCO2_v_r_np, ppO2_v_r_np) *
              (q_vin_r_np *
                  (cb_o2(ppCO2_v_r_np, ppO2_v_r_np) - cb_o2(ppCO2_at_r_np, ppO2_at_r_np))));
  // right ventricle O2
  f_np[47] =
      (1. / V_v_r_np) *
      pow((dcb_c_o2_dpp_c_o2(ppCO2_v_r_np, ppO2_v_r_np) * dcb_o2_dpp_o2(ppCO2_v_r_np, ppO2_v_r_np) -
              dcb_o2_dpp_c_o2(ppCO2_v_r_np, ppO2_v_r_np) *
                  dcb_c_o2_dpp_o2(ppCO2_v_r_np, ppO2_v_r_np)),
          -1.) *
      (dcb_c_o2_dpp_c_o2(ppCO2_v_r_np, ppO2_v_r_np) *
              (q_vin_r_np *
                  (cb_o2(ppCO2_v_r_np, ppO2_v_r_np) - cb_o2(ppCO2_at_r_np, ppO2_at_r_np))) -
          dcb_o2_dpp_c_o2(ppCO2_v_r_np, ppO2_v_r_np) *
              (q_vin_r_np *
                  (cb_c_o2(ppCO2_v_r_np, ppO2_v_r_np) - cb_c_o2(ppCO2_at_r_np, ppO2_at_r_np))));

  // pulmonary arteries CO2
  f_np[48] =
      (1. / V_ar_pul_np) *
      pow((dcb_c_o2_dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                  dcb_o2_dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) -
              dcb_o2_dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                  dcb_c_o2_dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np)),
          -1.) *
      (dcb_o2_dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
              (q_vout_r_np *
                  (cb_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) - cb_c_o2(ppCO2_v_r_np, ppO2_v_r_np))) -
          dcb_c_o2_dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
              (q_vout_r_np *
                  (cb_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) - cb_o2(ppCO2_v_r_np, ppO2_v_r_np))));
  // pulmonary arteries O2
  f_np[49] =
      (1. / V_ar_pul_np) *
      pow((dcb_c_o2_dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                  dcb_o2_dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) -
              dcb_o2_dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                  dcb_c_o2_dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np)),
          -1.) *
      (dcb_c_o2_dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
              (q_vout_r_np *
                  (cb_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) - cb_o2(ppCO2_v_r_np, ppO2_v_r_np))) -
          dcb_o2_dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
              (q_vout_r_np *
                  (cb_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) - cb_c_o2(ppCO2_v_r_np, ppO2_v_r_np))));

  // pulmonary capillaries CO2
  f_np[50] =
      (1. / V_cap_pul_np) *
      pow((dcb_c_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                  dcb_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) -
              dcb_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                  dcb_c_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np)),
          -1.) *
      (dcb_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
              (q_ar_pul_np * (cb_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) -
                                 cb_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np)) +
                  kappa_c_o2_ * (ppCO2_cap_pul_np - fCO2_alv_np * (p_alv_np - p_vap_water_37_))) -
          dcb_c_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
              (q_ar_pul_np * (cb_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) -
                                 cb_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np)) +
                  kappa_o2_ * (ppO2_cap_pul_np - fO2_alv_np * (p_alv_np - p_vap_water_37_))));
  // pulmonary capillaries O2
  f_np[51] =
      (1. / V_cap_pul_np) *
      pow((dcb_c_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                  dcb_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) -
              dcb_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                  dcb_c_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np)),
          -1.) *
      (dcb_c_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
              (q_ar_pul_np * (cb_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) -
                                 cb_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np)) +
                  kappa_o2_ * (ppO2_cap_pul_np - fO2_alv_np * (p_alv_np - p_vap_water_37_))) -
          dcb_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
              (q_ar_pul_np * (cb_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) -
                                 cb_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np)) +
                  kappa_c_o2_ * (ppCO2_cap_pul_np - fCO2_alv_np * (p_alv_np - p_vap_water_37_))));

  // pulmonary veins CO2
  f_np[52] = (1. / V_ven_pul_np) *
             pow((dcb_c_o2_dpp_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) *
                         dcb_o2_dpp_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) -
                     dcb_o2_dpp_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) *
                         dcb_c_o2_dpp_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np)),
                 -1.) *
             (dcb_o2_dpp_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) *
                     (q_cap_pul_np * (cb_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) -
                                         cb_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np))) -
                 dcb_c_o2_dpp_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) *
                     (q_cap_pul_np * (cb_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) -
                                         cb_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np))));
  // pulmonary veins O2
  f_np[53] = (1. / V_ven_pul_np) *
             pow((dcb_c_o2_dpp_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) *
                         dcb_o2_dpp_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) -
                     dcb_o2_dpp_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) *
                         dcb_c_o2_dpp_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np)),
                 -1.) *
             (dcb_c_o2_dpp_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) *
                     (q_cap_pul_np * (cb_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) -
                                         cb_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np))) -
                 dcb_o2_dpp_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) *
                     (q_cap_pul_np * (cb_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) -
                                         cb_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np))));

  // left atrium CO2
  f_np[54] = (1. / V_at_l_np) *
             pow((dcb_c_o2_dpp_c_o2(ppCO2_at_l_np, ppO2_at_l_np) *
                         dcb_o2_dpp_o2(ppCO2_at_l_np, ppO2_at_l_np) -
                     dcb_o2_dpp_c_o2(ppCO2_at_l_np, ppO2_at_l_np) *
                         dcb_c_o2_dpp_o2(ppCO2_at_l_np, ppO2_at_l_np)),
                 -1.) *
             (dcb_o2_dpp_o2(ppCO2_at_l_np, ppO2_at_l_np) *
                     (q_ven_pul_np * (cb_c_o2(ppCO2_at_l_np, ppO2_at_l_np) -
                                         cb_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np))) -
                 dcb_c_o2_dpp_o2(ppCO2_at_l_np, ppO2_at_l_np) *
                     (q_ven_pul_np * (cb_o2(ppCO2_at_l_np, ppO2_at_l_np) -
                                         cb_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np))));
  // left atrium O2
  f_np[55] = (1. / V_at_l_np) *
             pow((dcb_c_o2_dpp_c_o2(ppCO2_at_l_np, ppO2_at_l_np) *
                         dcb_o2_dpp_o2(ppCO2_at_l_np, ppO2_at_l_np) -
                     dcb_o2_dpp_c_o2(ppCO2_at_l_np, ppO2_at_l_np) *
                         dcb_c_o2_dpp_o2(ppCO2_at_l_np, ppO2_at_l_np)),
                 -1.) *
             (dcb_c_o2_dpp_c_o2(ppCO2_at_l_np, ppO2_at_l_np) *
                     (q_ven_pul_np * (cb_o2(ppCO2_at_l_np, ppO2_at_l_np) -
                                         cb_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np))) -
                 dcb_o2_dpp_c_o2(ppCO2_at_l_np, ppO2_at_l_np) *
                     (q_ven_pul_np * (cb_c_o2(ppCO2_at_l_np, ppO2_at_l_np) -
                                         cb_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np))));

  // left ventricle CO2
  f_np[56] =
      (1. / V_v_l_np) *
      pow((dcb_c_o2_dpp_c_o2(ppCO2_v_l_np, ppO2_v_l_np) * dcb_o2_dpp_o2(ppCO2_v_l_np, ppO2_v_l_np) -
              dcb_o2_dpp_c_o2(ppCO2_v_l_np, ppO2_v_l_np) *
                  dcb_c_o2_dpp_o2(ppCO2_v_l_np, ppO2_v_l_np)),
          -1.) *
      (dcb_o2_dpp_o2(ppCO2_v_l_np, ppO2_v_l_np) *
              (q_vin_l_np *
                  (cb_c_o2(ppCO2_v_l_np, ppO2_v_l_np) - cb_c_o2(ppCO2_at_l_np, ppO2_at_l_np))) -
          dcb_c_o2_dpp_o2(ppCO2_v_l_np, ppO2_v_l_np) *
              (q_vin_l_np *
                  (cb_o2(ppCO2_v_l_np, ppO2_v_l_np) - cb_o2(ppCO2_at_l_np, ppO2_at_l_np))));
  // left ventricle O2
  f_np[57] =
      (1. / V_v_l_np) *
      pow((dcb_c_o2_dpp_c_o2(ppCO2_v_l_np, ppO2_v_l_np) * dcb_o2_dpp_o2(ppCO2_v_l_np, ppO2_v_l_np) -
              dcb_o2_dpp_c_o2(ppCO2_v_l_np, ppO2_v_l_np) *
                  dcb_c_o2_dpp_o2(ppCO2_v_l_np, ppO2_v_l_np)),
          -1.) *
      (dcb_c_o2_dpp_c_o2(ppCO2_v_l_np, ppO2_v_l_np) *
              (q_vin_l_np *
                  (cb_o2(ppCO2_v_l_np, ppO2_v_l_np) - cb_o2(ppCO2_at_l_np, ppO2_at_l_np))) -
          dcb_o2_dpp_c_o2(ppCO2_v_l_np, ppO2_v_l_np) *
              (q_vin_l_np *
                  (cb_c_o2(ppCO2_v_l_np, ppO2_v_l_np) - cb_c_o2(ppCO2_at_l_np, ppO2_at_l_np))));

  // systemic arteries CO2
  f_np[58] =
      (1. / V_ar_sys_np) *
      pow((dcb_c_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                  dcb_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) -
              dcb_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                  dcb_c_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)),
          -1.) *
      (dcb_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
              (q_vout_l_np *
                  (cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) - cb_c_o2(ppCO2_v_l_np, ppO2_v_l_np))) -
          dcb_c_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
              (q_vout_l_np *
                  (cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) - cb_o2(ppCO2_v_l_np, ppO2_v_l_np))));
  // systemic arteries O2
  f_np[59] =
      (1. / V_ar_sys_np) *
      pow((dcb_c_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                  dcb_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) -
              dcb_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                  dcb_c_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)),
          -1.) *
      (dcb_c_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
              (q_vout_l_np *
                  (cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) - cb_o2(ppCO2_v_l_np, ppO2_v_l_np))) -
          dcb_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
              (q_vout_l_np *
                  (cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) - cb_c_o2(ppCO2_v_l_np, ppO2_v_l_np))));

  const double beta = 0.01;
  //// systemic capillaries
  // systemic splanchnic arteries CO2
  f_np[60] =
      (1. / V_arspl_sys_np) *
      pow(((dcb_c_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
               (v_tissspl_ / V_arspl_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arspl_sys_np)) *
                  (dcb_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
                      (v_tissspl_ / V_arspl_sys_np) * dct_o2_dpp_o2(ppO2_arspl_sys_np)) -
              dcb_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) *
                  dcb_c_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np)),
          -1.) *
      ((dcb_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
           (v_tissspl_ / V_arspl_sys_np) * dct_o2_dpp_o2(ppO2_arspl_sys_np)) *
              (q_arspl_sys_in_np * (cb_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) -
                                       cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                  m_c_o2_arspl_) -
          dcb_c_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) *
              (q_arspl_sys_in_np * (cb_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) -
                                       cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) +
                  m_o2_arspl_ * ct_o2(ppO2_arspl_sys_np) / (beta + ct_o2(ppO2_arspl_sys_np))));
  // systemic splanchnic arteries O2
  f_np[61] =
      (1. / V_arspl_sys_np) *
      pow(((dcb_c_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
               (v_tissspl_ / V_arspl_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arspl_sys_np)) *
                  (dcb_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
                      (v_tissspl_ / V_arspl_sys_np) * dct_o2_dpp_o2(ppO2_arspl_sys_np)) -
              dcb_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) *
                  dcb_c_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np)),
          -1.) *
      ((dcb_c_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
           (v_tissspl_ / V_arspl_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arspl_sys_np)) *
              (q_arspl_sys_in_np * (cb_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) -
                                       cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) +
                  m_o2_arspl_ * ct_o2(ppO2_arspl_sys_np) / (beta + ct_o2(ppO2_arspl_sys_np))) -
          dcb_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) *
              (q_arspl_sys_in_np * (cb_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) -
                                       cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                  m_c_o2_arspl_));

  // systemic extra-esplanchnic arteries CO2
  f_np[62] =
      (1. / V_arespl_sys_np) *
      pow(((dcb_c_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
               (v_tissespl_ / V_arespl_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arespl_sys_np)) *
                  (dcb_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
                      (v_tissespl_ / V_arespl_sys_np) * dct_o2_dpp_o2(ppO2_arespl_sys_np)) -
              dcb_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) *
                  dcb_c_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np)),
          -1.) *
      ((dcb_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
           (v_tissespl_ / V_arespl_sys_np) * dct_o2_dpp_o2(ppO2_arespl_sys_np)) *
              (q_arespl_sys_in_np * (cb_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) -
                                        cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                  m_c_o2_arespl_) -
          dcb_c_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) *
              (q_arespl_sys_in_np * (cb_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) -
                                        cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) +
                  m_o2_arespl_ * ct_o2(ppO2_arespl_sys_np) / (beta + ct_o2(ppO2_arespl_sys_np))));
  // systemic exrta-splanchnic arteries O2
  f_np[63] =
      (1. / V_arespl_sys_np) *
      pow(((dcb_c_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
               (v_tissespl_ / V_arespl_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arespl_sys_np)) *
                  (dcb_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
                      (v_tissespl_ / V_arespl_sys_np) * dct_o2_dpp_o2(ppO2_arespl_sys_np)) -
              dcb_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) *
                  dcb_c_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np)),
          -1.) *
      ((dcb_c_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
           (v_tissespl_ / V_arespl_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arespl_sys_np)) *
              (q_arespl_sys_in_np * (cb_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) -
                                        cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) +
                  m_o2_arespl_ * ct_o2(ppO2_arespl_sys_np) / (beta + ct_o2(ppO2_arespl_sys_np))) -
          dcb_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) *
              (q_arespl_sys_in_np * (cb_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) -
                                        cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                  m_c_o2_arespl_));

  // systemic muscular arteries CO2
  f_np[64] =
      (1. / V_armsc_sys_np) *
      pow(((dcb_c_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
               (v_tissmsc_ / V_armsc_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_armsc_sys_np)) *
                  (dcb_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
                      (v_tissmsc_ / V_armsc_sys_np) * dct_o2_dpp_o2(ppO2_armsc_sys_np)) -
              dcb_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) *
                  dcb_c_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np)),
          -1.) *
      ((dcb_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
           (v_tissmsc_ / V_armsc_sys_np) * dct_o2_dpp_o2(ppO2_armsc_sys_np)) *
              (q_armsc_sys_in_np * (cb_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) -
                                       cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                  m_c_o2_armsc_) -
          dcb_c_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) *
              (q_armsc_sys_in_np * (cb_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) -
                                       cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) +
                  m_o2_armsc_ * ct_o2(ppO2_armsc_sys_np) / (beta + ct_o2(ppO2_armsc_sys_np))));
  // systemic muscular arteries O2
  f_np[65] =
      (1. / V_armsc_sys_np) *
      pow(((dcb_c_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
               (v_tissmsc_ / V_armsc_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_armsc_sys_np)) *
                  (dcb_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
                      (v_tissmsc_ / V_armsc_sys_np) * dct_o2_dpp_o2(ppO2_armsc_sys_np)) -
              dcb_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) *
                  dcb_c_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np)),
          -1.) *
      ((dcb_c_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
           (v_tissmsc_ / V_armsc_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_armsc_sys_np)) *
              (q_armsc_sys_in_np * (cb_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) -
                                       cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) +
                  m_o2_armsc_ * ct_o2(ppO2_armsc_sys_np) / (beta + ct_o2(ppO2_armsc_sys_np))) -
          dcb_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) *
              (q_armsc_sys_in_np * (cb_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) -
                                       cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                  m_c_o2_armsc_));

  // systemic cerebral arteries CO2
  f_np[66] =
      (1. / V_arcer_sys_np) *
      pow(((dcb_c_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
               (v_tisscer_ / V_arcer_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arcer_sys_np)) *
                  (dcb_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
                      (v_tisscer_ / V_arcer_sys_np) * dct_o2_dpp_o2(ppO2_arcer_sys_np)) -
              dcb_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) *
                  dcb_c_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np)),
          -1.) *
      ((dcb_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
           (v_tisscer_ / V_arcer_sys_np) * dct_o2_dpp_o2(ppO2_arcer_sys_np)) *
              (q_arcer_sys_in_np * (cb_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) -
                                       cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                  m_c_o2_arcer_) -
          dcb_c_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) *
              (q_arcer_sys_in_np * (cb_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) -
                                       cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) +
                  m_o2_arcer_ * ct_o2(ppO2_arcer_sys_np) / (beta + ct_o2(ppO2_arcer_sys_np))));
  // systemic cerebral arteries O2
  f_np[67] =
      (1. / V_arcer_sys_np) *
      pow(((dcb_c_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
               (v_tisscer_ / V_arcer_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arcer_sys_np)) *
                  (dcb_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
                      (v_tisscer_ / V_arcer_sys_np) * dct_o2_dpp_o2(ppO2_arcer_sys_np)) -
              dcb_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) *
                  dcb_c_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np)),
          -1.) *
      ((dcb_c_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
           (v_tisscer_ / V_arcer_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arcer_sys_np)) *
              (q_arcer_sys_in_np * (cb_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) -
                                       cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) +
                  m_o2_arcer_ * ct_o2(ppO2_arcer_sys_np) / (beta + ct_o2(ppO2_arcer_sys_np))) -
          dcb_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) *
              (q_arcer_sys_in_np * (cb_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) -
                                       cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                  m_c_o2_arcer_));

  // systemic coronary arteries CO2
  f_np[68] =
      (1. / V_arcor_sys_np) *
      pow(((dcb_c_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
               (v_tisscor_ / V_arcor_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arcor_sys_np)) *
                  (dcb_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
                      (v_tisscor_ / V_arcor_sys_np) * dct_o2_dpp_o2(ppO2_arcor_sys_np)) -
              dcb_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) *
                  dcb_c_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np)),
          -1.) *
      ((dcb_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
           (v_tisscor_ / V_arcor_sys_np) * dct_o2_dpp_o2(ppO2_arcor_sys_np)) *
              (q_arcor_sys_in_np * (cb_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) -
                                       cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                  m_c_o2_arcor_) -
          dcb_c_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) *
              (q_arcor_sys_in_np * (cb_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) -
                                       cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) +
                  m_o2_arcor_ * ct_o2(ppO2_arcor_sys_np) / (beta + ct_o2(ppO2_arcor_sys_np))));
  // systemic coronary arteries O2
  f_np[69] =
      (1. / V_arcor_sys_np) *
      pow(((dcb_c_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
               (v_tisscor_ / V_arcor_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arcor_sys_np)) *
                  (dcb_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
                      (v_tisscor_ / V_arcor_sys_np) * dct_o2_dpp_o2(ppO2_arcor_sys_np)) -
              dcb_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) *
                  dcb_c_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np)),
          -1.) *
      ((dcb_c_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
           (v_tisscor_ / V_arcor_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arcor_sys_np)) *
              (q_arcor_sys_in_np * (cb_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) -
                                       cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) +
                  m_o2_arcor_ * ct_o2(ppO2_arcor_sys_np) / (beta + ct_o2(ppO2_arcor_sys_np))) -
          dcb_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) *
              (q_arcor_sys_in_np * (cb_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) -
                                       cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                  m_c_o2_arcor_));


  // systemic splanchnic veins CO2
  f_np[70] = (1. / V_venspl_sys_np) *
             pow((dcb_c_o2_dpp_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                         dcb_o2_dpp_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) -
                     dcb_o2_dpp_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                         dcb_c_o2_dpp_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np)),
                 -1.) *
             (dcb_o2_dpp_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                     (q_arspl_sys_np * (cb_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) -
                                           cb_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np))) -
                 dcb_c_o2_dpp_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                     (q_arspl_sys_np * (cb_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) -
                                           cb_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np))));
  // systemic splanchnic veins O2
  f_np[71] = (1. / V_venspl_sys_np) *
             pow((dcb_c_o2_dpp_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                         dcb_o2_dpp_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) -
                     dcb_o2_dpp_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                         dcb_c_o2_dpp_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np)),
                 -1.) *
             (dcb_c_o2_dpp_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                     (q_arspl_sys_np * (cb_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) -
                                           cb_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np))) -
                 dcb_o2_dpp_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                     (q_arspl_sys_np * (cb_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) -
                                           cb_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np))));

  // systemic extra-splanchnic veins CO2
  f_np[72] = (1. / V_venespl_sys_np) *
             pow((dcb_c_o2_dpp_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                         dcb_o2_dpp_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) -
                     dcb_o2_dpp_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                         dcb_c_o2_dpp_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np)),
                 -1.) *
             (dcb_o2_dpp_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                     (q_arespl_sys_np * (cb_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) -
                                            cb_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np))) -
                 dcb_c_o2_dpp_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                     (q_arespl_sys_np * (cb_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) -
                                            cb_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np))));
  // systemic extra-splanchnic veins O2
  f_np[73] = (1. / V_venespl_sys_np) *
             pow((dcb_c_o2_dpp_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                         dcb_o2_dpp_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) -
                     dcb_o2_dpp_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                         dcb_c_o2_dpp_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np)),
                 -1.) *
             (dcb_c_o2_dpp_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                     (q_arespl_sys_np * (cb_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) -
                                            cb_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np))) -
                 dcb_o2_dpp_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                     (q_arespl_sys_np * (cb_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) -
                                            cb_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np))));

  // systemic muscular veins CO2
  f_np[74] = (1. / V_venmsc_sys_np) *
             pow((dcb_c_o2_dpp_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                         dcb_o2_dpp_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) -
                     dcb_o2_dpp_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                         dcb_c_o2_dpp_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np)),
                 -1.) *
             (dcb_o2_dpp_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                     (q_armsc_sys_np * (cb_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) -
                                           cb_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np))) -
                 dcb_c_o2_dpp_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                     (q_armsc_sys_np * (cb_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) -
                                           cb_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np))));
  // systemic muscular veins O2
  f_np[75] = (1. / V_venmsc_sys_np) *
             pow((dcb_c_o2_dpp_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                         dcb_o2_dpp_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) -
                     dcb_o2_dpp_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                         dcb_c_o2_dpp_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np)),
                 -1.) *
             (dcb_c_o2_dpp_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                     (q_armsc_sys_np * (cb_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) -
                                           cb_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np))) -
                 dcb_o2_dpp_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                     (q_armsc_sys_np * (cb_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) -
                                           cb_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np))));

  // systemic cerebral veins CO2
  f_np[76] = (1. / V_vencer_sys_np) *
             pow((dcb_c_o2_dpp_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                         dcb_o2_dpp_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) -
                     dcb_o2_dpp_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                         dcb_c_o2_dpp_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np)),
                 -1.) *
             (dcb_o2_dpp_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                     (q_arcer_sys_np * (cb_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) -
                                           cb_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np))) -
                 dcb_c_o2_dpp_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                     (q_arcer_sys_np * (cb_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) -
                                           cb_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np))));
  // systemic cerebral veins O2
  f_np[77] = (1. / V_vencer_sys_np) *
             pow((dcb_c_o2_dpp_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                         dcb_o2_dpp_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) -
                     dcb_o2_dpp_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                         dcb_c_o2_dpp_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np)),
                 -1.) *
             (dcb_c_o2_dpp_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                     (q_arcer_sys_np * (cb_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) -
                                           cb_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np))) -
                 dcb_o2_dpp_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                     (q_arcer_sys_np * (cb_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) -
                                           cb_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np))));

  // systemic coronary veins CO2
  f_np[78] = (1. / V_vencor_sys_np) *
             pow((dcb_c_o2_dpp_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                         dcb_o2_dpp_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) -
                     dcb_o2_dpp_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                         dcb_c_o2_dpp_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np)),
                 -1.) *
             (dcb_o2_dpp_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                     (q_arcor_sys_np * (cb_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) -
                                           cb_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np))) -
                 dcb_c_o2_dpp_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                     (q_arcor_sys_np * (cb_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) -
                                           cb_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np))));
  // systemic coronary veins O2
  f_np[79] = (1. / V_vencor_sys_np) *
             pow((dcb_c_o2_dpp_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                         dcb_o2_dpp_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) -
                     dcb_o2_dpp_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                         dcb_c_o2_dpp_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np)),
                 -1.) *
             (dcb_c_o2_dpp_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                     (q_arcor_sys_np * (cb_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) -
                                           cb_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np))) -
                 dcb_o2_dpp_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                     (q_arcor_sys_np * (cb_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) -
                                           cb_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np))));

  // mixture rule for joining flows: c_upstr = (q_upstr_1 * c_upstr_1 + ... + q_upstr_n * c_upstr_n)
  // / (q_upstr_1 + ... + q_upstr_n) systemic veins CO2
  f_np[80] = (1. / V_ven_sys_np) *
             pow((dcb_c_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                         dcb_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                     dcb_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                         dcb_c_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np)),
                 -1.) *
             (dcb_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                     (((q_venspl_sys_np + q_venespl_sys_np + q_venmsc_sys_np + q_vencer_sys_np +
                           q_vencor_sys_np) *
                             cb_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                         (q_venspl_sys_np * cb_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) +
                             q_venespl_sys_np * cb_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) +
                             q_venmsc_sys_np * cb_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) +
                             q_vencer_sys_np * cb_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) +
                             q_vencor_sys_np * cb_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np)))) -
                 dcb_c_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                     (((q_venspl_sys_np + q_venespl_sys_np + q_venmsc_sys_np + q_vencer_sys_np +
                           q_vencor_sys_np) *
                             cb_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                         (q_venspl_sys_np * cb_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) +
                             q_venespl_sys_np * cb_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) +
                             q_venmsc_sys_np * cb_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) +
                             q_vencer_sys_np * cb_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) +
                             q_vencor_sys_np * cb_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np)))));
  // systemic veins O2
  f_np[81] = (1. / V_ven_sys_np) *
             pow((dcb_c_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                         dcb_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                     dcb_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                         dcb_c_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np)),
                 -1.) *
             (dcb_c_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                     (((q_venspl_sys_np + q_venespl_sys_np + q_venmsc_sys_np + q_vencer_sys_np +
                           q_vencor_sys_np) *
                             cb_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                         (q_venspl_sys_np * cb_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) +
                             q_venespl_sys_np * cb_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) +
                             q_venmsc_sys_np * cb_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) +
                             q_vencer_sys_np * cb_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) +
                             q_vencor_sys_np * cb_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np)))) -
                 dcb_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                     (((q_venspl_sys_np + q_venespl_sys_np + q_venmsc_sys_np + q_vencer_sys_np +
                           q_vencor_sys_np) *
                             cb_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                         (q_venspl_sys_np * cb_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) +
                             q_venespl_sys_np * cb_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) +
                             q_venmsc_sys_np * cb_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) +
                             q_vencer_sys_np * cb_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) +
                             q_vencor_sys_np * cb_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np)))));

  // contributions to stiffness
  if (evalstiff)
  {
    wkstiff(34, 34) = 1. / ts_size;
    wkstiff(34, 35) = -theta;

    wkstiff(35, 34) = theta * e_alv_;
    wkstiff(35, 35) = l_alv_ / ts_size + theta * r_alv_;
    wkstiff(35, 36) = -theta;

    wkstiff(36, 34) =
        theta *
        ((1. / (V_alv_np * V_alv_np)) *
            (u_m_ * ((u_m_ - p_alv_np) / r_alv_ +
                        v_m_gas_ * kappa_c_o2_ *
                            (ppCO2_cap_pul_np - fCO2_alv_np * (p_alv_np - p_vap_water_37_)) +
                        v_m_gas_ * kappa_o2_ *
                            (ppO2_cap_pul_np - fO2_alv_np * (p_alv_np - p_vap_water_37_))) -
                p_alv_np * q_alv_np));
    wkstiff(36, 35) = theta * ((1. / V_alv_np) * p_alv_np);
    wkstiff(36, 36) =
        1. / ts_size +
        theta *
            (-(1. / V_alv_np) * (u_m_ * ((-1.) / r_alv_ + v_m_gas_ * kappa_c_o2_ * (-fCO2_alv_np) +
                                            v_m_gas_ * kappa_o2_ * (-fO2_alv_np)) -
                                    q_alv_np));
    wkstiff(36, 37) =
        theta * ((1. / V_alv_np) * u_m_ * v_m_gas_ * kappa_c_o2_ * (p_alv_np - p_vap_water_37_));
    wkstiff(36, 38) =
        theta * ((1. / V_alv_np) * u_m_ * v_m_gas_ * kappa_o2_ * (p_alv_np - p_vap_water_37_));
    wkstiff(36, 50) = theta * (-(1. / V_alv_np) * u_m_ * v_m_gas_ * kappa_c_o2_);
    wkstiff(36, 51) = theta * (-(1. / V_alv_np) * u_m_ * v_m_gas_ * kappa_o2_);

    double dfCO2_insp = 0.;
    double dfO2_insp = 0.;

    if (v_lung_tidal_ >= v_lung_dead_) dfCO2_insp = v_lung_dead_ / v_lung_tidal_;
    if (v_lung_tidal_ < v_lung_dead_) dfCO2_insp = 1.0;

    if (v_lung_tidal_ >= v_lung_dead_) dfO2_insp = v_lung_dead_ / v_lung_tidal_;
    if (v_lung_tidal_ < v_lung_dead_) dfO2_insp = 1.0;

    double dq_insp = 0.;

    if ((u_m_ - p_alv_np) / r_alv_ > 0.) dq_insp = -1. / r_alv_;
    if ((u_m_ - p_alv_np) / r_alv_ <= 0.) dq_insp = 0.;

    wkstiff(37, 34) =
        theta *
        ((1. / (V_alv_np * V_alv_np)) *
            (v_m_gas_ * kappa_c_o2_ * (ppCO2_cap_pul_np - fCO2_alv_np * p_alv_np) +
                (fCO2_insp - fCO2_alv_np) * q_insp -
                fCO2_alv_np *
                    (v_m_gas_ * kappa_o2_ *
                            (ppO2_cap_pul_np - fO2_alv_np * (p_alv_np - p_vap_water_37_)) +
                        v_m_gas_ * kappa_c_o2_ *
                            (ppCO2_cap_pul_np - fCO2_alv_np * (p_alv_np - p_vap_water_37_)))));
    wkstiff(37, 36) =
        theta * (-(1. / V_alv_np) *
                    (v_m_gas_ * kappa_c_o2_ * (-fCO2_alv_np) + (fCO2_insp - fCO2_alv_np) * dq_insp -
                        fCO2_alv_np * (v_m_gas_ * kappa_o2_ * (-fO2_alv_np) +
                                          v_m_gas_ * kappa_c_o2_ * (-fCO2_alv_np))));
    wkstiff(37, 37) =
        1. / ts_size +
        theta *
            (-(1. / V_alv_np) *
                (v_m_gas_ * kappa_c_o2_ * (-(p_alv_np - p_vap_water_37_)) +
                    (dfCO2_insp - 1.) * q_insp -
                    1. * (v_m_gas_ * kappa_o2_ *
                                 (ppO2_cap_pul_np - fO2_alv_np * (p_alv_np - p_vap_water_37_)) +
                             v_m_gas_ * kappa_c_o2_ *
                                 (ppCO2_cap_pul_np - fCO2_alv_np * (p_alv_np - p_vap_water_37_))) -
                    fCO2_alv_np * (v_m_gas_ * kappa_c_o2_ * (-(p_alv_np - p_vap_water_37_)))));
    wkstiff(37, 38) =
        theta * (-(1. / V_alv_np) *
                    (-fCO2_alv_np * (v_m_gas_ * kappa_o2_ * (-(p_alv_np - p_vap_water_37_)))));
    wkstiff(37, 50) =
        theta * (-(1. / V_alv_np) * (v_m_gas_ * kappa_c_o2_ * (1.) -
                                        fCO2_alv_np * (v_m_gas_ * kappa_c_o2_ * (1.))));
    wkstiff(37, 51) = theta * (-(1. / V_alv_np) * (-fCO2_alv_np * (v_m_gas_ * kappa_o2_ * (1.))));

    wkstiff(38, 34) =
        theta *
        ((1. / (V_alv_np * V_alv_np)) *
            (v_m_gas_ * kappa_o2_ * (ppO2_cap_pul_np - fO2_alv_np * (p_alv_np - p_vap_water_37_)) +
                (fO2_insp - fO2_alv_np) * q_insp -
                fO2_alv_np *
                    (v_m_gas_ * kappa_c_o2_ *
                            (ppCO2_cap_pul_np - fCO2_alv_np * (p_alv_np - p_vap_water_37_)) +
                        v_m_gas_ * kappa_o2_ *
                            (ppO2_cap_pul_np - fO2_alv_np * (p_alv_np - p_vap_water_37_)))));
    wkstiff(38, 36) =
        theta * (-(1. / V_alv_np) *
                    (v_m_gas_ * kappa_o2_ * (-fO2_alv_np) + (fO2_insp - fO2_alv_np) * dq_insp -
                        fO2_alv_np * (v_m_gas_ * kappa_c_o2_ * (-fCO2_alv_np) +
                                         v_m_gas_ * kappa_o2_ * (-fO2_alv_np))));
    wkstiff(38, 37) =
        theta * (-(1. / V_alv_np) *
                    (-fO2_alv_np * (v_m_gas_ * kappa_c_o2_ * (-(p_alv_np - p_vap_water_37_)))));
    wkstiff(38, 38) =
        1. / ts_size +
        theta *
            (-(1. / V_alv_np) *
                (v_m_gas_ * kappa_o2_ * (-(p_alv_np - p_vap_water_37_)) +
                    (dfO2_insp - 1.) * q_insp -
                    1. * (v_m_gas_ * kappa_c_o2_ *
                                 (ppCO2_cap_pul_np - fCO2_alv_np * (p_alv_np - p_vap_water_37_)) +
                             v_m_gas_ * kappa_o2_ *
                                 (ppO2_cap_pul_np - fO2_alv_np * (p_alv_np - p_vap_water_37_))) -
                    fO2_alv_np * (v_m_gas_ * kappa_o2_ * (-(p_alv_np - p_vap_water_37_)))));
    wkstiff(38, 50) = theta * (-(1. / V_alv_np) * (-fO2_alv_np * (v_m_gas_ * kappa_c_o2_ * (1.))));
    wkstiff(38, 51) = theta * (-(1. / V_alv_np) * (v_m_gas_ * kappa_o2_ * (1.) -
                                                      fO2_alv_np * (v_m_gas_ * kappa_o2_ * (1.))));

    // since we need the derivative of atrial and ventricular volumes w.r.t. to pressures, we have
    // to check what type of model we have
    double dV_at_l_dp = 0.;
    double dV_at_r_dp = 0.;
    double dV_v_l_dp = 0.;
    double dV_v_r_dp = 0.;

    switch (atrium_model_)
    {
      case Inpar::Cardiovascular0D::atr_elastance_0d:
      {
        dV_at_l_dp = df_np[0] / p_at_l_np;
        dV_at_r_dp = df_np[24] / p_at_r_np;
      }
      break;
      case Inpar::Cardiovascular0D::atr_structure_3d:
      {
        dV_at_l_dp = 0.;
        dV_at_r_dp = 0.;
      }
      break;
      case Inpar::Cardiovascular0D::atr_prescribed:
      {
        dV_at_l_dp = df_np[0] / p_at_l_np;
        dV_at_r_dp = df_np[24] / p_at_r_np;
      }
      break;
      default:
        FOUR_C_THROW("Undefined atrium_model!");
        break;
    }

    switch (ventricle_model_)
    {
      case Inpar::Cardiovascular0D::ventr_structure_3d:
      {
        dV_v_l_dp = 0.;
        dV_v_r_dp = 0.;
      }
      break;
      case Inpar::Cardiovascular0D::ventr_elastance_0d:
      {
        dV_v_l_dp = df_np[2] / p_v_l_np;
        dV_v_r_dp = df_np[26] / p_v_r_np;
      }
      break;
      case Inpar::Cardiovascular0D::ventr_prescribed:
      {
        dV_v_l_dp = df_np[2] / p_v_l_np;
        dV_v_r_dp = df_np[26] / p_v_r_np;
      }
      break;
      default:
        FOUR_C_THROW("Undefined ventricle_model!");
        break;
    }

    wkstiff(39, 6) = c_arspl_sys_ / ts_size;
    wkstiff(39, 7) = theta;
    wkstiff(39, 39) = -theta;

    wkstiff(40, 6) = c_arespl_sys_ / ts_size;
    wkstiff(40, 8) = theta;
    wkstiff(40, 40) = -theta;

    wkstiff(41, 6) = c_armsc_sys_ / ts_size;
    wkstiff(41, 9) = theta;
    wkstiff(41, 41) = -theta;

    wkstiff(42, 6) = c_arcer_sys_ / ts_size;
    wkstiff(42, 10) = theta;
    wkstiff(42, 42) = -theta;

    wkstiff(43, 6) = c_arcor_sys_ / ts_size;
    wkstiff(43, 11) = theta;
    wkstiff(43, 43) = -theta;

    //////// right atrium CO2
    // w.r.t. upstream flux
    wkstiff(44, 23) = theta * ((1. / V_at_r_np) *
                                  pow((dcb_c_o2_dpp_c_o2(ppCO2_at_r_np, ppO2_at_r_np) *
                                              dcb_o2_dpp_o2(ppCO2_at_r_np, ppO2_at_r_np) -
                                          dcb_o2_dpp_c_o2(ppCO2_at_r_np, ppO2_at_r_np) *
                                              dcb_c_o2_dpp_o2(ppCO2_at_r_np, ppO2_at_r_np)),
                                      -1.) *
                                  (dcb_o2_dpp_o2(ppCO2_at_r_np, ppO2_at_r_np) *
                                          (1.0 * (cb_c_o2(ppCO2_at_r_np, ppO2_at_r_np) -
                                                     cb_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np))) -
                                      dcb_c_o2_dpp_o2(ppCO2_at_r_np, ppO2_at_r_np) *
                                          (1.0 * (cb_o2(ppCO2_at_r_np, ppO2_at_r_np) -
                                                     cb_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np)))));
    // w.r.t. mech. pressure
    wkstiff(44, 24) =
        theta * (dV_at_r_dp * (-1. / (V_at_r_np * V_at_r_np)) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_at_r_np, ppO2_at_r_np) *
                                dcb_o2_dpp_o2(ppCO2_at_r_np, ppO2_at_r_np) -
                            dcb_o2_dpp_c_o2(ppCO2_at_r_np, ppO2_at_r_np) *
                                dcb_c_o2_dpp_o2(ppCO2_at_r_np, ppO2_at_r_np)),
                        -1.) *
                    (dcb_o2_dpp_o2(ppCO2_at_r_np, ppO2_at_r_np) *
                            (q_ven_sys_np * (cb_c_o2(ppCO2_at_r_np, ppO2_at_r_np) -
                                                cb_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np))) -
                        dcb_c_o2_dpp_o2(ppCO2_at_r_np, ppO2_at_r_np) *
                            (q_ven_sys_np * (cb_o2(ppCO2_at_r_np, ppO2_at_r_np) -
                                                cb_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np)))));
    // w.r.t. ppCO2
    wkstiff(44, 44) =
        1. / ts_size +
        theta * (-(1. / V_at_r_np) *
                        pow((dcb_c_o2_dpp_c_o2(ppCO2_at_r_np, ppO2_at_r_np) *
                                    dcb_o2_dpp_o2(ppCO2_at_r_np, ppO2_at_r_np) -
                                dcb_o2_dpp_c_o2(ppCO2_at_r_np, ppO2_at_r_np) *
                                    dcb_c_o2_dpp_o2(ppCO2_at_r_np, ppO2_at_r_np)),
                            -2.) *
                        (d2cb_c_o2_dpp_c_o22(ppCO2_at_r_np, ppO2_at_r_np) *
                                dcb_o2_dpp_o2(ppCO2_at_r_np, ppO2_at_r_np) +
                            dcb_c_o2_dpp_c_o2(ppCO2_at_r_np, ppO2_at_r_np) *
                                d2cb_o2_dpp_o2dpp_c_o2(ppCO2_at_r_np, ppO2_at_r_np) -
                            d2cb_o2_dpp_c_o22(ppCO2_at_r_np, ppO2_at_r_np) *
                                dcb_c_o2_dpp_o2(ppCO2_at_r_np, ppO2_at_r_np) -
                            dcb_o2_dpp_c_o2(ppCO2_at_r_np, ppO2_at_r_np) *
                                d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_at_r_np, ppO2_at_r_np)) *
                        (dcb_o2_dpp_o2(ppCO2_at_r_np, ppO2_at_r_np) *
                                (q_ven_sys_np * (cb_c_o2(ppCO2_at_r_np, ppO2_at_r_np) -
                                                    cb_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np))) -
                            dcb_c_o2_dpp_o2(ppCO2_at_r_np, ppO2_at_r_np) *
                                (q_ven_sys_np * (cb_o2(ppCO2_at_r_np, ppO2_at_r_np) -
                                                    cb_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np)))) +
                    (1. / V_at_r_np) *
                        pow((dcb_c_o2_dpp_c_o2(ppCO2_at_r_np, ppO2_at_r_np) *
                                    dcb_o2_dpp_o2(ppCO2_at_r_np, ppO2_at_r_np) -
                                dcb_o2_dpp_c_o2(ppCO2_at_r_np, ppO2_at_r_np) *
                                    dcb_c_o2_dpp_o2(ppCO2_at_r_np, ppO2_at_r_np)),
                            -1.) *
                        (d2cb_o2_dpp_o2dpp_c_o2(ppCO2_at_r_np, ppO2_at_r_np) *
                                (q_ven_sys_np * (cb_c_o2(ppCO2_at_r_np, ppO2_at_r_np) -
                                                    cb_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np))) +
                            dcb_o2_dpp_o2(ppCO2_at_r_np, ppO2_at_r_np) * q_ven_sys_np *
                                dcb_c_o2_dpp_c_o2(ppCO2_at_r_np, ppO2_at_r_np) -
                            d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_at_r_np, ppO2_at_r_np) *
                                (q_ven_sys_np * (cb_o2(ppCO2_at_r_np, ppO2_at_r_np) -
                                                    cb_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np))) -
                            dcb_c_o2_dpp_o2(ppCO2_at_r_np, ppO2_at_r_np) * q_ven_sys_np *
                                dcb_o2_dpp_c_o2(ppCO2_at_r_np, ppO2_at_r_np)));
    // w.r.t. ppO2
    wkstiff(44, 45) =
        theta * (-(1. / V_at_r_np) *
                        pow((dcb_c_o2_dpp_c_o2(ppCO2_at_r_np, ppO2_at_r_np) *
                                    dcb_o2_dpp_o2(ppCO2_at_r_np, ppO2_at_r_np) -
                                dcb_o2_dpp_c_o2(ppCO2_at_r_np, ppO2_at_r_np) *
                                    dcb_c_o2_dpp_o2(ppCO2_at_r_np, ppO2_at_r_np)),
                            -2.) *
                        (d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_at_r_np, ppO2_at_r_np) *
                                dcb_o2_dpp_o2(ppCO2_at_r_np, ppO2_at_r_np) +
                            dcb_c_o2_dpp_c_o2(ppCO2_at_r_np, ppO2_at_r_np) *
                                d2cb_o2_dpp_o22(ppCO2_at_r_np, ppO2_at_r_np) -
                            d2cb_o2_dpp_o2dpp_c_o2(ppCO2_at_r_np, ppO2_at_r_np) *
                                dcb_c_o2_dpp_o2(ppCO2_at_r_np, ppO2_at_r_np) -
                            dcb_o2_dpp_c_o2(ppCO2_at_r_np, ppO2_at_r_np) *
                                d2cb_c_o2_dpp_o22(ppCO2_at_r_np, ppO2_at_r_np)) *
                        (dcb_o2_dpp_o2(ppCO2_at_r_np, ppO2_at_r_np) *
                                (q_ven_sys_np * (cb_c_o2(ppCO2_at_r_np, ppO2_at_r_np) -
                                                    cb_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np))) -
                            dcb_c_o2_dpp_o2(ppCO2_at_r_np, ppO2_at_r_np) *
                                (q_ven_sys_np * (cb_o2(ppCO2_at_r_np, ppO2_at_r_np) -
                                                    cb_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np)))) +
                    (1. / V_at_r_np) *
                        pow((dcb_c_o2_dpp_c_o2(ppCO2_at_r_np, ppO2_at_r_np) *
                                    dcb_o2_dpp_o2(ppCO2_at_r_np, ppO2_at_r_np) -
                                dcb_o2_dpp_c_o2(ppCO2_at_r_np, ppO2_at_r_np) *
                                    dcb_c_o2_dpp_o2(ppCO2_at_r_np, ppO2_at_r_np)),
                            -1.) *
                        (d2cb_o2_dpp_o22(ppCO2_at_r_np, ppO2_at_r_np) *
                                (q_ven_sys_np * (cb_c_o2(ppCO2_at_r_np, ppO2_at_r_np) -
                                                    cb_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np))) +
                            dcb_o2_dpp_o2(ppCO2_at_r_np, ppO2_at_r_np) * q_ven_sys_np *
                                dcb_c_o2_dpp_o2(ppCO2_at_r_np, ppO2_at_r_np) -
                            d2cb_c_o2_dpp_o22(ppCO2_at_r_np, ppO2_at_r_np) *
                                (q_ven_sys_np * (cb_o2(ppCO2_at_r_np, ppO2_at_r_np) -
                                                    cb_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np))) -
                            dcb_c_o2_dpp_o2(ppCO2_at_r_np, ppO2_at_r_np) * q_ven_sys_np *
                                dcb_o2_dpp_o2(ppCO2_at_r_np, ppO2_at_r_np)));
    // w.r.t. upstream ppCO2
    wkstiff(44, 80) = theta * (-(1. / V_at_r_np) *
                                  pow((dcb_c_o2_dpp_c_o2(ppCO2_at_r_np, ppO2_at_r_np) *
                                              dcb_o2_dpp_o2(ppCO2_at_r_np, ppO2_at_r_np) -
                                          dcb_o2_dpp_c_o2(ppCO2_at_r_np, ppO2_at_r_np) *
                                              dcb_c_o2_dpp_o2(ppCO2_at_r_np, ppO2_at_r_np)),
                                      -1.) *
                                  (dcb_o2_dpp_o2(ppCO2_at_r_np, ppO2_at_r_np) * q_ven_sys_np *
                                          dcb_c_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                                      dcb_c_o2_dpp_o2(ppCO2_at_r_np, ppO2_at_r_np) * q_ven_sys_np *
                                          dcb_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np)));
    // w.r.t. upstream ppO2
    wkstiff(44, 81) = theta * (-(1. / V_at_r_np) *
                                  pow((dcb_c_o2_dpp_c_o2(ppCO2_at_r_np, ppO2_at_r_np) *
                                              dcb_o2_dpp_o2(ppCO2_at_r_np, ppO2_at_r_np) -
                                          dcb_o2_dpp_c_o2(ppCO2_at_r_np, ppO2_at_r_np) *
                                              dcb_c_o2_dpp_o2(ppCO2_at_r_np, ppO2_at_r_np)),
                                      -1.) *
                                  (dcb_o2_dpp_o2(ppCO2_at_r_np, ppO2_at_r_np) * q_ven_sys_np *
                                          dcb_c_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                                      dcb_c_o2_dpp_o2(ppCO2_at_r_np, ppO2_at_r_np) * q_ven_sys_np *
                                          dcb_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np)));

    //////// right atrium O2
    // w.r.t. upstream flux
    wkstiff(45, 23) = theta * ((1. / V_at_r_np) *
                                  pow((dcb_c_o2_dpp_c_o2(ppCO2_at_r_np, ppO2_at_r_np) *
                                              dcb_o2_dpp_o2(ppCO2_at_r_np, ppO2_at_r_np) -
                                          dcb_o2_dpp_c_o2(ppCO2_at_r_np, ppO2_at_r_np) *
                                              dcb_c_o2_dpp_o2(ppCO2_at_r_np, ppO2_at_r_np)),
                                      -1.) *
                                  (dcb_c_o2_dpp_c_o2(ppCO2_at_r_np, ppO2_at_r_np) *
                                          (1.0 * (cb_o2(ppCO2_at_r_np, ppO2_at_r_np) -
                                                     cb_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np))) -
                                      dcb_o2_dpp_c_o2(ppCO2_at_r_np, ppO2_at_r_np) *
                                          (1.0 * (cb_c_o2(ppCO2_at_r_np, ppO2_at_r_np) -
                                                     cb_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np)))));
    // w.r.t. mech. pressure
    wkstiff(45, 24) =
        theta * (dV_at_r_dp * (-1. / (V_at_r_np * V_at_r_np)) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_at_r_np, ppO2_at_r_np) *
                                dcb_o2_dpp_o2(ppCO2_at_r_np, ppO2_at_r_np) -
                            dcb_o2_dpp_c_o2(ppCO2_at_r_np, ppO2_at_r_np) *
                                dcb_c_o2_dpp_o2(ppCO2_at_r_np, ppO2_at_r_np)),
                        -1.) *
                    (dcb_c_o2_dpp_c_o2(ppCO2_at_r_np, ppO2_at_r_np) *
                            (q_ven_sys_np * (cb_o2(ppCO2_at_r_np, ppO2_at_r_np) -
                                                cb_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np))) -
                        dcb_o2_dpp_c_o2(ppCO2_at_r_np, ppO2_at_r_np) *
                            (q_ven_sys_np * (cb_c_o2(ppCO2_at_r_np, ppO2_at_r_np) -
                                                cb_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np)))));
    // w.r.t. ppCO2
    wkstiff(45, 44) =
        theta * (-(1. / V_at_r_np) *
                        pow((dcb_c_o2_dpp_c_o2(ppCO2_at_r_np, ppO2_at_r_np) *
                                    dcb_o2_dpp_o2(ppCO2_at_r_np, ppO2_at_r_np) -
                                dcb_o2_dpp_c_o2(ppCO2_at_r_np, ppO2_at_r_np) *
                                    dcb_c_o2_dpp_o2(ppCO2_at_r_np, ppO2_at_r_np)),
                            -2.) *
                        (d2cb_c_o2_dpp_c_o22(ppCO2_at_r_np, ppO2_at_r_np) *
                                dcb_o2_dpp_o2(ppCO2_at_r_np, ppO2_at_r_np) +
                            dcb_c_o2_dpp_c_o2(ppCO2_at_r_np, ppO2_at_r_np) *
                                d2cb_o2_dpp_o2dpp_c_o2(ppCO2_at_r_np, ppO2_at_r_np) -
                            d2cb_o2_dpp_c_o22(ppCO2_at_r_np, ppO2_at_r_np) *
                                dcb_c_o2_dpp_o2(ppCO2_at_r_np, ppO2_at_r_np) -
                            dcb_o2_dpp_c_o2(ppCO2_at_r_np, ppO2_at_r_np) *
                                d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_at_r_np, ppO2_at_r_np)) *
                        (dcb_c_o2_dpp_c_o2(ppCO2_at_r_np, ppO2_at_r_np) *
                                (q_ven_sys_np * (cb_o2(ppCO2_at_r_np, ppO2_at_r_np) -
                                                    cb_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np))) -
                            dcb_o2_dpp_c_o2(ppCO2_at_r_np, ppO2_at_r_np) *
                                (q_ven_sys_np * (cb_c_o2(ppCO2_at_r_np, ppO2_at_r_np) -
                                                    cb_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np)))) +
                    (1. / V_at_r_np) *
                        pow((dcb_c_o2_dpp_c_o2(ppCO2_at_r_np, ppO2_at_r_np) *
                                    dcb_o2_dpp_o2(ppCO2_at_r_np, ppO2_at_r_np) -
                                dcb_o2_dpp_c_o2(ppCO2_at_r_np, ppO2_at_r_np) *
                                    dcb_c_o2_dpp_o2(ppCO2_at_r_np, ppO2_at_r_np)),
                            -1.) *
                        (d2cb_c_o2_dpp_c_o22(ppCO2_at_r_np, ppO2_at_r_np) *
                                (q_ven_sys_np * (cb_o2(ppCO2_at_r_np, ppO2_at_r_np) -
                                                    cb_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np))) +
                            dcb_c_o2_dpp_c_o2(ppCO2_at_r_np, ppO2_at_r_np) * q_ven_sys_np *
                                dcb_o2_dpp_c_o2(ppCO2_at_r_np, ppO2_at_r_np) -
                            d2cb_o2_dpp_c_o22(ppCO2_at_r_np, ppO2_at_r_np) *
                                (q_ven_sys_np * (cb_c_o2(ppCO2_at_r_np, ppO2_at_r_np) -
                                                    cb_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np))) -
                            dcb_o2_dpp_c_o2(ppCO2_at_r_np, ppO2_at_r_np) * q_ven_sys_np *
                                dcb_c_o2_dpp_c_o2(ppCO2_at_r_np, ppO2_at_r_np)));
    // w.r.t. ppO2
    wkstiff(45, 45) =
        1. / ts_size +
        theta * (-(1. / V_at_r_np) *
                        pow((dcb_c_o2_dpp_c_o2(ppCO2_at_r_np, ppO2_at_r_np) *
                                    dcb_o2_dpp_o2(ppCO2_at_r_np, ppO2_at_r_np) -
                                dcb_o2_dpp_c_o2(ppCO2_at_r_np, ppO2_at_r_np) *
                                    dcb_c_o2_dpp_o2(ppCO2_at_r_np, ppO2_at_r_np)),
                            -2.) *
                        (d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_at_r_np, ppO2_at_r_np) *
                                dcb_o2_dpp_o2(ppCO2_at_r_np, ppO2_at_r_np) +
                            dcb_c_o2_dpp_c_o2(ppCO2_at_r_np, ppO2_at_r_np) *
                                d2cb_o2_dpp_o22(ppCO2_at_r_np, ppO2_at_r_np) -
                            d2cb_o2_dpp_o2dpp_c_o2(ppCO2_at_r_np, ppO2_at_r_np) *
                                dcb_c_o2_dpp_o2(ppCO2_at_r_np, ppO2_at_r_np) -
                            dcb_o2_dpp_c_o2(ppCO2_at_r_np, ppO2_at_r_np) *
                                d2cb_c_o2_dpp_o22(ppCO2_at_r_np, ppO2_at_r_np)) *
                        (dcb_c_o2_dpp_c_o2(ppCO2_at_r_np, ppO2_at_r_np) *
                                (q_ven_sys_np * (cb_o2(ppCO2_at_r_np, ppO2_at_r_np) -
                                                    cb_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np))) -
                            dcb_o2_dpp_c_o2(ppCO2_at_r_np, ppO2_at_r_np) *
                                (q_ven_sys_np * (cb_c_o2(ppCO2_at_r_np, ppO2_at_r_np) -
                                                    cb_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np)))) +
                    (1. / V_at_r_np) *
                        pow((dcb_c_o2_dpp_c_o2(ppCO2_at_r_np, ppO2_at_r_np) *
                                    dcb_o2_dpp_o2(ppCO2_at_r_np, ppO2_at_r_np) -
                                dcb_o2_dpp_c_o2(ppCO2_at_r_np, ppO2_at_r_np) *
                                    dcb_c_o2_dpp_o2(ppCO2_at_r_np, ppO2_at_r_np)),
                            -1.) *
                        (d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_at_r_np, ppO2_at_r_np) *
                                (q_ven_sys_np * (cb_o2(ppCO2_at_r_np, ppO2_at_r_np) -
                                                    cb_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np))) +
                            dcb_c_o2_dpp_c_o2(ppCO2_at_r_np, ppO2_at_r_np) * q_ven_sys_np *
                                dcb_o2_dpp_o2(ppCO2_at_r_np, ppO2_at_r_np) -
                            d2cb_o2_dpp_o2dpp_c_o2(ppCO2_at_r_np, ppO2_at_r_np) *
                                (q_ven_sys_np * (cb_c_o2(ppCO2_at_r_np, ppO2_at_r_np) -
                                                    cb_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np))) -
                            dcb_o2_dpp_c_o2(ppCO2_at_r_np, ppO2_at_r_np) * q_ven_sys_np *
                                dcb_c_o2_dpp_o2(ppCO2_at_r_np, ppO2_at_r_np)));
    // w.r.t. upstream ppCO2
    wkstiff(45, 80) = theta * (-(1. / V_at_r_np) *
                                  pow((dcb_c_o2_dpp_c_o2(ppCO2_at_r_np, ppO2_at_r_np) *
                                              dcb_o2_dpp_o2(ppCO2_at_r_np, ppO2_at_r_np) -
                                          dcb_o2_dpp_c_o2(ppCO2_at_r_np, ppO2_at_r_np) *
                                              dcb_c_o2_dpp_o2(ppCO2_at_r_np, ppO2_at_r_np)),
                                      -1.) *
                                  (dcb_c_o2_dpp_c_o2(ppCO2_at_r_np, ppO2_at_r_np) * q_ven_sys_np *
                                          dcb_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                                      dcb_o2_dpp_c_o2(ppCO2_at_r_np, ppO2_at_r_np) * q_ven_sys_np *
                                          dcb_c_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np)));
    // w.r.t. upstream ppO2
    wkstiff(45, 81) = theta * (-(1. / V_at_r_np) *
                                  pow((dcb_c_o2_dpp_c_o2(ppCO2_at_r_np, ppO2_at_r_np) *
                                              dcb_o2_dpp_o2(ppCO2_at_r_np, ppO2_at_r_np) -
                                          dcb_o2_dpp_c_o2(ppCO2_at_r_np, ppO2_at_r_np) *
                                              dcb_c_o2_dpp_o2(ppCO2_at_r_np, ppO2_at_r_np)),
                                      -1.) *
                                  (dcb_c_o2_dpp_c_o2(ppCO2_at_r_np, ppO2_at_r_np) * q_ven_sys_np *
                                          dcb_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                                      dcb_o2_dpp_c_o2(ppCO2_at_r_np, ppO2_at_r_np) * q_ven_sys_np *
                                          dcb_c_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np)));


    //////// right ventricle CO2
    // w.r.t. upstream flux
    wkstiff(46, 25) = theta * ((1. / V_v_r_np) *
                                  pow((dcb_c_o2_dpp_c_o2(ppCO2_v_r_np, ppO2_v_r_np) *
                                              dcb_o2_dpp_o2(ppCO2_v_r_np, ppO2_v_r_np) -
                                          dcb_o2_dpp_c_o2(ppCO2_v_r_np, ppO2_v_r_np) *
                                              dcb_c_o2_dpp_o2(ppCO2_v_r_np, ppO2_v_r_np)),
                                      -1.) *
                                  (dcb_o2_dpp_o2(ppCO2_v_r_np, ppO2_v_r_np) *
                                          (1.0 * (cb_c_o2(ppCO2_v_r_np, ppO2_v_r_np) -
                                                     cb_c_o2(ppCO2_at_r_np, ppO2_at_r_np))) -
                                      dcb_c_o2_dpp_o2(ppCO2_v_r_np, ppO2_v_r_np) *
                                          (1.0 * (cb_o2(ppCO2_v_r_np, ppO2_v_r_np) -
                                                     cb_o2(ppCO2_at_r_np, ppO2_at_r_np)))));
    // w.r.t. mech. pressure
    wkstiff(46, 27) = theta * (dV_v_r_dp * (-1. / (V_v_r_np * V_v_r_np)) *
                                  pow((dcb_c_o2_dpp_c_o2(ppCO2_v_r_np, ppO2_v_r_np) *
                                              dcb_o2_dpp_o2(ppCO2_v_r_np, ppO2_v_r_np) -
                                          dcb_o2_dpp_c_o2(ppCO2_v_r_np, ppO2_v_r_np) *
                                              dcb_c_o2_dpp_o2(ppCO2_v_r_np, ppO2_v_r_np)),
                                      -1.) *
                                  (dcb_o2_dpp_o2(ppCO2_v_r_np, ppO2_v_r_np) *
                                          (q_vin_r_np * (cb_c_o2(ppCO2_v_r_np, ppO2_v_r_np) -
                                                            cb_c_o2(ppCO2_at_r_np, ppO2_at_r_np))) -
                                      dcb_c_o2_dpp_o2(ppCO2_v_r_np, ppO2_v_r_np) *
                                          (q_vin_r_np * (cb_o2(ppCO2_v_r_np, ppO2_v_r_np) -
                                                            cb_o2(ppCO2_at_r_np, ppO2_at_r_np)))));
    // w.r.t. upstream ppCO2
    wkstiff(46, 44) = theta * (-(1. / V_v_r_np) *
                                  pow((dcb_c_o2_dpp_c_o2(ppCO2_v_r_np, ppO2_v_r_np) *
                                              dcb_o2_dpp_o2(ppCO2_v_r_np, ppO2_v_r_np) -
                                          dcb_o2_dpp_c_o2(ppCO2_v_r_np, ppO2_v_r_np) *
                                              dcb_c_o2_dpp_o2(ppCO2_v_r_np, ppO2_v_r_np)),
                                      -1.) *
                                  (dcb_o2_dpp_o2(ppCO2_v_r_np, ppO2_v_r_np) * q_vin_r_np *
                                          dcb_c_o2_dpp_c_o2(ppCO2_at_r_np, ppO2_at_r_np) -
                                      dcb_c_o2_dpp_o2(ppCO2_v_r_np, ppO2_v_r_np) * q_vin_r_np *
                                          dcb_o2_dpp_c_o2(ppCO2_at_r_np, ppO2_at_r_np)));
    // w.r.t. upstream ppO2
    wkstiff(46, 45) = theta * (-(1. / V_v_r_np) *
                                  pow((dcb_c_o2_dpp_c_o2(ppCO2_v_r_np, ppO2_v_r_np) *
                                              dcb_o2_dpp_o2(ppCO2_v_r_np, ppO2_v_r_np) -
                                          dcb_o2_dpp_c_o2(ppCO2_v_r_np, ppO2_v_r_np) *
                                              dcb_c_o2_dpp_o2(ppCO2_v_r_np, ppO2_v_r_np)),
                                      -1.) *
                                  (dcb_o2_dpp_o2(ppCO2_v_r_np, ppO2_v_r_np) * q_vin_r_np *
                                          dcb_c_o2_dpp_o2(ppCO2_at_r_np, ppO2_at_r_np) -
                                      dcb_c_o2_dpp_o2(ppCO2_v_r_np, ppO2_v_r_np) * q_vin_r_np *
                                          dcb_o2_dpp_o2(ppCO2_at_r_np, ppO2_at_r_np)));
    // w.r.t. ppCO2
    wkstiff(46, 46) =
        1. / ts_size +
        theta * (-(1. / V_v_r_np) *
                        pow((dcb_c_o2_dpp_c_o2(ppCO2_v_r_np, ppO2_v_r_np) *
                                    dcb_o2_dpp_o2(ppCO2_v_r_np, ppO2_v_r_np) -
                                dcb_o2_dpp_c_o2(ppCO2_v_r_np, ppO2_v_r_np) *
                                    dcb_c_o2_dpp_o2(ppCO2_v_r_np, ppO2_v_r_np)),
                            -2.) *
                        (d2cb_c_o2_dpp_c_o22(ppCO2_v_r_np, ppO2_v_r_np) *
                                dcb_o2_dpp_o2(ppCO2_v_r_np, ppO2_v_r_np) +
                            dcb_c_o2_dpp_c_o2(ppCO2_v_r_np, ppO2_v_r_np) *
                                d2cb_o2_dpp_o2dpp_c_o2(ppCO2_v_r_np, ppO2_v_r_np) -
                            d2cb_o2_dpp_c_o22(ppCO2_v_r_np, ppO2_v_r_np) *
                                dcb_c_o2_dpp_o2(ppCO2_v_r_np, ppO2_v_r_np) -
                            dcb_o2_dpp_c_o2(ppCO2_v_r_np, ppO2_v_r_np) *
                                d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_v_r_np, ppO2_v_r_np)) *
                        (dcb_o2_dpp_o2(ppCO2_v_r_np, ppO2_v_r_np) *
                                (q_vin_r_np * (cb_c_o2(ppCO2_v_r_np, ppO2_v_r_np) -
                                                  cb_c_o2(ppCO2_at_r_np, ppO2_at_r_np))) -
                            dcb_c_o2_dpp_o2(ppCO2_v_r_np, ppO2_v_r_np) *
                                (q_vin_r_np * (cb_o2(ppCO2_v_r_np, ppO2_v_r_np) -
                                                  cb_o2(ppCO2_at_r_np, ppO2_at_r_np)))) +
                    (1. / V_v_r_np) *
                        pow((dcb_c_o2_dpp_c_o2(ppCO2_v_r_np, ppO2_v_r_np) *
                                    dcb_o2_dpp_o2(ppCO2_v_r_np, ppO2_v_r_np) -
                                dcb_o2_dpp_c_o2(ppCO2_v_r_np, ppO2_v_r_np) *
                                    dcb_c_o2_dpp_o2(ppCO2_v_r_np, ppO2_v_r_np)),
                            -1.) *
                        (d2cb_o2_dpp_o2dpp_c_o2(ppCO2_v_r_np, ppO2_v_r_np) *
                                (q_vin_r_np * (cb_c_o2(ppCO2_v_r_np, ppO2_v_r_np) -
                                                  cb_c_o2(ppCO2_at_r_np, ppO2_at_r_np))) +
                            dcb_o2_dpp_o2(ppCO2_v_r_np, ppO2_v_r_np) * q_vin_r_np *
                                dcb_c_o2_dpp_c_o2(ppCO2_v_r_np, ppO2_v_r_np) -
                            d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_v_r_np, ppO2_v_r_np) *
                                (q_vin_r_np * (cb_o2(ppCO2_v_r_np, ppO2_v_r_np) -
                                                  cb_o2(ppCO2_at_r_np, ppO2_at_r_np))) -
                            dcb_c_o2_dpp_o2(ppCO2_v_r_np, ppO2_v_r_np) * q_vin_r_np *
                                dcb_o2_dpp_c_o2(ppCO2_v_r_np, ppO2_v_r_np)));
    // w.r.t. ppO2
    wkstiff(46, 47) =
        theta * (-(1. / V_v_r_np) *
                        pow((dcb_c_o2_dpp_c_o2(ppCO2_v_r_np, ppO2_v_r_np) *
                                    dcb_o2_dpp_o2(ppCO2_v_r_np, ppO2_v_r_np) -
                                dcb_o2_dpp_c_o2(ppCO2_v_r_np, ppO2_v_r_np) *
                                    dcb_c_o2_dpp_o2(ppCO2_v_r_np, ppO2_v_r_np)),
                            -2.) *
                        (d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_v_r_np, ppO2_v_r_np) *
                                dcb_o2_dpp_o2(ppCO2_v_r_np, ppO2_v_r_np) +
                            dcb_c_o2_dpp_c_o2(ppCO2_v_r_np, ppO2_v_r_np) *
                                d2cb_o2_dpp_o22(ppCO2_v_r_np, ppO2_v_r_np) -
                            d2cb_o2_dpp_o2dpp_c_o2(ppCO2_v_r_np, ppO2_v_r_np) *
                                dcb_c_o2_dpp_o2(ppCO2_v_r_np, ppO2_v_r_np) -
                            dcb_o2_dpp_c_o2(ppCO2_v_r_np, ppO2_v_r_np) *
                                d2cb_c_o2_dpp_o22(ppCO2_v_r_np, ppO2_v_r_np)) *
                        (dcb_o2_dpp_o2(ppCO2_v_r_np, ppO2_v_r_np) *
                                (q_vin_r_np * (cb_c_o2(ppCO2_v_r_np, ppO2_v_r_np) -
                                                  cb_c_o2(ppCO2_at_r_np, ppO2_at_r_np))) -
                            dcb_c_o2_dpp_o2(ppCO2_v_r_np, ppO2_v_r_np) *
                                (q_vin_r_np * (cb_o2(ppCO2_v_r_np, ppO2_v_r_np) -
                                                  cb_o2(ppCO2_at_r_np, ppO2_at_r_np)))) +
                    (1. / V_v_r_np) *
                        pow((dcb_c_o2_dpp_c_o2(ppCO2_v_r_np, ppO2_v_r_np) *
                                    dcb_o2_dpp_o2(ppCO2_v_r_np, ppO2_v_r_np) -
                                dcb_o2_dpp_c_o2(ppCO2_v_r_np, ppO2_v_r_np) *
                                    dcb_c_o2_dpp_o2(ppCO2_v_r_np, ppO2_v_r_np)),
                            -1.) *
                        (d2cb_o2_dpp_o22(ppCO2_v_r_np, ppO2_v_r_np) *
                                (q_vin_r_np * (cb_c_o2(ppCO2_v_r_np, ppO2_v_r_np) -
                                                  cb_c_o2(ppCO2_at_r_np, ppO2_at_r_np))) +
                            dcb_o2_dpp_o2(ppCO2_v_r_np, ppO2_v_r_np) * q_vin_r_np *
                                dcb_c_o2_dpp_o2(ppCO2_v_r_np, ppO2_v_r_np) -
                            d2cb_c_o2_dpp_o22(ppCO2_v_r_np, ppO2_v_r_np) *
                                (q_vin_r_np * (cb_o2(ppCO2_v_r_np, ppO2_v_r_np) -
                                                  cb_o2(ppCO2_at_r_np, ppO2_at_r_np))) -
                            dcb_c_o2_dpp_o2(ppCO2_v_r_np, ppO2_v_r_np) * q_vin_r_np *
                                dcb_o2_dpp_o2(ppCO2_v_r_np, ppO2_v_r_np)));

    //////// right ventricle O2
    // w.r.t. upstream flux
    wkstiff(47, 25) = theta * ((1. / V_v_r_np) *
                                  pow((dcb_c_o2_dpp_c_o2(ppCO2_v_r_np, ppO2_v_r_np) *
                                              dcb_o2_dpp_o2(ppCO2_v_r_np, ppO2_v_r_np) -
                                          dcb_o2_dpp_c_o2(ppCO2_v_r_np, ppO2_v_r_np) *
                                              dcb_c_o2_dpp_o2(ppCO2_v_r_np, ppO2_v_r_np)),
                                      -1.) *
                                  (dcb_c_o2_dpp_c_o2(ppCO2_v_r_np, ppO2_v_r_np) *
                                          (1.0 * (cb_o2(ppCO2_v_r_np, ppO2_v_r_np) -
                                                     cb_o2(ppCO2_at_r_np, ppO2_at_r_np))) -
                                      dcb_o2_dpp_c_o2(ppCO2_v_r_np, ppO2_v_r_np) *
                                          (1.0 * (cb_c_o2(ppCO2_v_r_np, ppO2_v_r_np) -
                                                     cb_c_o2(ppCO2_at_r_np, ppO2_at_r_np)))));
    // w.r.t. mech. pressure
    wkstiff(47, 27) =
        theta * (dV_v_r_dp * (-1. / (V_v_r_np * V_v_r_np)) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_v_r_np, ppO2_v_r_np) *
                                dcb_o2_dpp_o2(ppCO2_v_r_np, ppO2_v_r_np) -
                            dcb_o2_dpp_c_o2(ppCO2_v_r_np, ppO2_v_r_np) *
                                dcb_c_o2_dpp_o2(ppCO2_v_r_np, ppO2_v_r_np)),
                        -1.) *
                    (dcb_c_o2_dpp_c_o2(ppCO2_v_r_np, ppO2_v_r_np) *
                            (q_vin_r_np * (cb_o2(ppCO2_v_r_np, ppO2_v_r_np) -
                                              cb_o2(ppCO2_at_r_np, ppO2_at_r_np))) -
                        dcb_o2_dpp_c_o2(ppCO2_v_r_np, ppO2_v_r_np) *
                            (q_vin_r_np * (cb_c_o2(ppCO2_v_r_np, ppO2_v_r_np) -
                                              cb_c_o2(ppCO2_at_r_np, ppO2_at_r_np)))));
    // w.r.t. upstream ppCO2
    wkstiff(47, 44) = theta * (-(1. / V_v_r_np) *
                                  pow((dcb_c_o2_dpp_c_o2(ppCO2_v_r_np, ppO2_v_r_np) *
                                              dcb_o2_dpp_o2(ppCO2_v_r_np, ppO2_v_r_np) -
                                          dcb_o2_dpp_c_o2(ppCO2_v_r_np, ppO2_v_r_np) *
                                              dcb_c_o2_dpp_o2(ppCO2_v_r_np, ppO2_v_r_np)),
                                      -1.) *
                                  (dcb_c_o2_dpp_c_o2(ppCO2_v_r_np, ppO2_v_r_np) * q_vin_r_np *
                                          dcb_o2_dpp_c_o2(ppCO2_at_r_np, ppO2_at_r_np) -
                                      dcb_o2_dpp_c_o2(ppCO2_v_r_np, ppO2_v_r_np) * q_vin_r_np *
                                          dcb_c_o2_dpp_c_o2(ppCO2_at_r_np, ppO2_at_r_np)));
    // w.r.t. upstream ppO2
    wkstiff(47, 45) = theta * (-(1. / V_v_r_np) *
                                  pow((dcb_c_o2_dpp_c_o2(ppCO2_v_r_np, ppO2_v_r_np) *
                                              dcb_o2_dpp_o2(ppCO2_v_r_np, ppO2_v_r_np) -
                                          dcb_o2_dpp_c_o2(ppCO2_v_r_np, ppO2_v_r_np) *
                                              dcb_c_o2_dpp_o2(ppCO2_v_r_np, ppO2_v_r_np)),
                                      -1.) *
                                  (dcb_c_o2_dpp_c_o2(ppCO2_v_r_np, ppO2_v_r_np) * q_vin_r_np *
                                          dcb_o2_dpp_o2(ppCO2_at_r_np, ppO2_at_r_np) -
                                      dcb_o2_dpp_c_o2(ppCO2_v_r_np, ppO2_v_r_np) * q_vin_r_np *
                                          dcb_c_o2_dpp_o2(ppCO2_at_r_np, ppO2_at_r_np)));
    // w.r.t. ppCO2
    wkstiff(47, 46) =
        theta * (-(1. / V_v_r_np) *
                        pow((dcb_c_o2_dpp_c_o2(ppCO2_v_r_np, ppO2_v_r_np) *
                                    dcb_o2_dpp_o2(ppCO2_v_r_np, ppO2_v_r_np) -
                                dcb_o2_dpp_c_o2(ppCO2_v_r_np, ppO2_v_r_np) *
                                    dcb_c_o2_dpp_o2(ppCO2_v_r_np, ppO2_v_r_np)),
                            -2.) *
                        (d2cb_c_o2_dpp_c_o22(ppCO2_v_r_np, ppO2_v_r_np) *
                                dcb_o2_dpp_o2(ppCO2_v_r_np, ppO2_v_r_np) +
                            dcb_c_o2_dpp_c_o2(ppCO2_v_r_np, ppO2_v_r_np) *
                                d2cb_o2_dpp_o2dpp_c_o2(ppCO2_v_r_np, ppO2_v_r_np) -
                            d2cb_o2_dpp_c_o22(ppCO2_v_r_np, ppO2_v_r_np) *
                                dcb_c_o2_dpp_o2(ppCO2_v_r_np, ppO2_v_r_np) -
                            dcb_o2_dpp_c_o2(ppCO2_v_r_np, ppO2_v_r_np) *
                                d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_v_r_np, ppO2_v_r_np)) *
                        (dcb_c_o2_dpp_c_o2(ppCO2_v_r_np, ppO2_v_r_np) *
                                (q_vin_r_np * (cb_o2(ppCO2_v_r_np, ppO2_v_r_np) -
                                                  cb_o2(ppCO2_at_r_np, ppO2_at_r_np))) -
                            dcb_o2_dpp_c_o2(ppCO2_v_r_np, ppO2_v_r_np) *
                                (q_vin_r_np * (cb_c_o2(ppCO2_v_r_np, ppO2_v_r_np) -
                                                  cb_c_o2(ppCO2_at_r_np, ppO2_at_r_np)))) +
                    (1. / V_v_r_np) *
                        pow((dcb_c_o2_dpp_c_o2(ppCO2_v_r_np, ppO2_v_r_np) *
                                    dcb_o2_dpp_o2(ppCO2_v_r_np, ppO2_v_r_np) -
                                dcb_o2_dpp_c_o2(ppCO2_v_r_np, ppO2_v_r_np) *
                                    dcb_c_o2_dpp_o2(ppCO2_v_r_np, ppO2_v_r_np)),
                            -1.) *
                        (d2cb_c_o2_dpp_c_o22(ppCO2_v_r_np, ppO2_v_r_np) *
                                (q_vin_r_np * (cb_o2(ppCO2_v_r_np, ppO2_v_r_np) -
                                                  cb_o2(ppCO2_at_r_np, ppO2_at_r_np))) +
                            dcb_c_o2_dpp_c_o2(ppCO2_v_r_np, ppO2_v_r_np) * q_vin_r_np *
                                dcb_o2_dpp_c_o2(ppCO2_v_r_np, ppO2_v_r_np) -
                            d2cb_o2_dpp_c_o22(ppCO2_v_r_np, ppO2_v_r_np) *
                                (q_vin_r_np * (cb_c_o2(ppCO2_v_r_np, ppO2_v_r_np) -
                                                  cb_c_o2(ppCO2_at_r_np, ppO2_at_r_np))) -
                            dcb_o2_dpp_c_o2(ppCO2_v_r_np, ppO2_v_r_np) * q_vin_r_np *
                                dcb_c_o2_dpp_c_o2(ppCO2_v_r_np, ppO2_v_r_np)));
    // w.r.t. ppO2
    wkstiff(47, 47) =
        1. / ts_size +
        theta * (-(1. / V_v_r_np) *
                        pow((dcb_c_o2_dpp_c_o2(ppCO2_v_r_np, ppO2_v_r_np) *
                                    dcb_o2_dpp_o2(ppCO2_v_r_np, ppO2_v_r_np) -
                                dcb_o2_dpp_c_o2(ppCO2_v_r_np, ppO2_v_r_np) *
                                    dcb_c_o2_dpp_o2(ppCO2_v_r_np, ppO2_v_r_np)),
                            -2.) *
                        (d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_v_r_np, ppO2_v_r_np) *
                                dcb_o2_dpp_o2(ppCO2_v_r_np, ppO2_v_r_np) +
                            dcb_c_o2_dpp_c_o2(ppCO2_v_r_np, ppO2_v_r_np) *
                                d2cb_o2_dpp_o22(ppCO2_v_r_np, ppO2_v_r_np) -
                            d2cb_o2_dpp_o2dpp_c_o2(ppCO2_v_r_np, ppO2_v_r_np) *
                                dcb_c_o2_dpp_o2(ppCO2_v_r_np, ppO2_v_r_np) -
                            dcb_o2_dpp_c_o2(ppCO2_v_r_np, ppO2_v_r_np) *
                                d2cb_c_o2_dpp_o22(ppCO2_v_r_np, ppO2_v_r_np)) *
                        (dcb_c_o2_dpp_c_o2(ppCO2_v_r_np, ppO2_v_r_np) *
                                (q_vin_r_np * (cb_o2(ppCO2_v_r_np, ppO2_v_r_np) -
                                                  cb_o2(ppCO2_at_r_np, ppO2_at_r_np))) -
                            dcb_o2_dpp_c_o2(ppCO2_v_r_np, ppO2_v_r_np) *
                                (q_vin_r_np * (cb_c_o2(ppCO2_v_r_np, ppO2_v_r_np) -
                                                  cb_c_o2(ppCO2_at_r_np, ppO2_at_r_np)))) +
                    (1. / V_v_r_np) *
                        pow((dcb_c_o2_dpp_c_o2(ppCO2_v_r_np, ppO2_v_r_np) *
                                    dcb_o2_dpp_o2(ppCO2_v_r_np, ppO2_v_r_np) -
                                dcb_o2_dpp_c_o2(ppCO2_v_r_np, ppO2_v_r_np) *
                                    dcb_c_o2_dpp_o2(ppCO2_v_r_np, ppO2_v_r_np)),
                            -1.) *
                        (d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_v_r_np, ppO2_v_r_np) *
                                (q_vin_r_np * (cb_o2(ppCO2_v_r_np, ppO2_v_r_np) -
                                                  cb_o2(ppCO2_at_r_np, ppO2_at_r_np))) +
                            dcb_c_o2_dpp_c_o2(ppCO2_v_r_np, ppO2_v_r_np) * q_vin_r_np *
                                dcb_o2_dpp_o2(ppCO2_v_r_np, ppO2_v_r_np) -
                            d2cb_o2_dpp_o2dpp_c_o2(ppCO2_v_r_np, ppO2_v_r_np) *
                                (q_vin_r_np * (cb_c_o2(ppCO2_v_r_np, ppO2_v_r_np) -
                                                  cb_c_o2(ppCO2_at_r_np, ppO2_at_r_np))) -
                            dcb_o2_dpp_c_o2(ppCO2_v_r_np, ppO2_v_r_np) * q_vin_r_np *
                                dcb_c_o2_dpp_o2(ppCO2_v_r_np, ppO2_v_r_np)));


    //////// pulmonary arteries CO2
    // w.r.t. mech. pressure
    wkstiff(48, 28) = theta * (c_ar_pul_ * (-1. / (V_ar_pul_np * V_ar_pul_np)) *
                                  pow((dcb_c_o2_dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                              dcb_o2_dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) -
                                          dcb_o2_dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                              dcb_c_o2_dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np)),
                                      -1.) *
                                  (dcb_o2_dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                          (q_vout_r_np * (cb_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) -
                                                             cb_c_o2(ppCO2_v_r_np, ppO2_v_r_np))) -
                                      dcb_c_o2_dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                          (q_vout_r_np * (cb_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) -
                                                             cb_o2(ppCO2_v_r_np, ppO2_v_r_np)))));
    // w.r.t. upstream flux
    wkstiff(48, 29) =
        theta * ((1. / V_ar_pul_np) *
                        pow((dcb_c_o2_dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                    dcb_o2_dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) -
                                dcb_o2_dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                    dcb_c_o2_dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np)),
                            -1.) *
                        (dcb_o2_dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                (1.0 * (cb_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) -
                                           cb_c_o2(ppCO2_v_r_np, ppO2_v_r_np))) -
                            dcb_c_o2_dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                (1.0 * (cb_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) -
                                           cb_o2(ppCO2_v_r_np, ppO2_v_r_np)))) +
                    (-c_ar_pul_ * z_ar_pul_) * (-1. / (V_ar_pul_np * V_ar_pul_np)) *
                        pow((dcb_c_o2_dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                    dcb_o2_dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) -
                                dcb_o2_dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                    dcb_c_o2_dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np)),
                            -1.) *
                        (dcb_o2_dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                (q_vout_r_np * (cb_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) -
                                                   cb_c_o2(ppCO2_v_r_np, ppO2_v_r_np))) -
                            dcb_c_o2_dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                (q_vout_r_np * (cb_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) -
                                                   cb_o2(ppCO2_v_r_np, ppO2_v_r_np)))));
    // w.r.t. upstream ppCO2
    wkstiff(48, 46) =
        theta * (-(1. / V_ar_pul_np) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                dcb_o2_dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) -
                            dcb_o2_dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                dcb_c_o2_dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np)),
                        -1.) *
                    (dcb_o2_dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) * q_vout_r_np *
                            dcb_c_o2_dpp_c_o2(ppCO2_v_r_np, ppO2_v_r_np) -
                        dcb_c_o2_dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) * q_vout_r_np *
                            dcb_o2_dpp_c_o2(ppCO2_v_r_np, ppO2_v_r_np)));
    // w.r.t. upstream ppO2
    wkstiff(48, 47) = theta * (-(1. / V_ar_pul_np) *
                                  pow((dcb_c_o2_dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                              dcb_o2_dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) -
                                          dcb_o2_dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                              dcb_c_o2_dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np)),
                                      -1.) *
                                  (dcb_o2_dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) * q_vout_r_np *
                                          dcb_c_o2_dpp_o2(ppCO2_v_r_np, ppO2_v_r_np) -
                                      dcb_c_o2_dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                          q_vout_r_np * dcb_o2_dpp_o2(ppCO2_v_r_np, ppO2_v_r_np)));
    // w.r.t. ppCO2
    wkstiff(48, 48) =
        1. / ts_size +
        theta * (-(1. / V_ar_pul_np) *
                        pow((dcb_c_o2_dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                    dcb_o2_dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) -
                                dcb_o2_dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                    dcb_c_o2_dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np)),
                            -2.) *
                        (d2cb_c_o2_dpp_c_o22(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                dcb_o2_dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) +
                            dcb_c_o2_dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                d2cb_o2_dpp_o2dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) -
                            d2cb_o2_dpp_c_o22(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                dcb_c_o2_dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) -
                            dcb_o2_dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np)) *
                        (dcb_o2_dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                (q_vout_r_np * (cb_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) -
                                                   cb_c_o2(ppCO2_v_r_np, ppO2_v_r_np))) -
                            dcb_c_o2_dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                (q_vout_r_np * (cb_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) -
                                                   cb_o2(ppCO2_v_r_np, ppO2_v_r_np)))) +
                    (1. / V_ar_pul_np) *
                        pow((dcb_c_o2_dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                    dcb_o2_dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) -
                                dcb_o2_dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                    dcb_c_o2_dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np)),
                            -1.) *
                        (d2cb_o2_dpp_o2dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                (q_vout_r_np * (cb_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) -
                                                   cb_c_o2(ppCO2_v_r_np, ppO2_v_r_np))) +
                            dcb_o2_dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) * q_vout_r_np *
                                dcb_c_o2_dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) -
                            d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                (q_vout_r_np * (cb_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) -
                                                   cb_o2(ppCO2_v_r_np, ppO2_v_r_np))) -
                            dcb_c_o2_dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) * q_vout_r_np *
                                dcb_o2_dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np)));
    // w.r.t. ppO2
    wkstiff(48, 49) =
        theta * (-(1. / V_ar_pul_np) *
                        pow((dcb_c_o2_dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                    dcb_o2_dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) -
                                dcb_o2_dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                    dcb_c_o2_dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np)),
                            -2.) *
                        (d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                dcb_o2_dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) +
                            dcb_c_o2_dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                d2cb_o2_dpp_o22(ppCO2_ar_pul_np, ppO2_ar_pul_np) -
                            d2cb_o2_dpp_o2dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                dcb_c_o2_dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) -
                            dcb_o2_dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                d2cb_c_o2_dpp_o22(ppCO2_ar_pul_np, ppO2_ar_pul_np)) *
                        (dcb_o2_dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                (q_vout_r_np * (cb_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) -
                                                   cb_c_o2(ppCO2_v_r_np, ppO2_v_r_np))) -
                            dcb_c_o2_dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                (q_vout_r_np * (cb_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) -
                                                   cb_o2(ppCO2_v_r_np, ppO2_v_r_np)))) +
                    (1. / V_ar_pul_np) *
                        pow((dcb_c_o2_dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                    dcb_o2_dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) -
                                dcb_o2_dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                    dcb_c_o2_dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np)),
                            -1.) *
                        (d2cb_o2_dpp_o22(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                (q_vout_r_np * (cb_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) -
                                                   cb_c_o2(ppCO2_v_r_np, ppO2_v_r_np))) +
                            dcb_o2_dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) * q_vout_r_np *
                                dcb_c_o2_dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) -
                            d2cb_c_o2_dpp_o22(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                (q_vout_r_np * (cb_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) -
                                                   cb_o2(ppCO2_v_r_np, ppO2_v_r_np))) -
                            dcb_c_o2_dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) * q_vout_r_np *
                                dcb_o2_dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np)));

    //////// pulmonary arteries O2
    // w.r.t. mech. pressure
    wkstiff(49, 28) = theta * (c_ar_pul_ * (-1. / (V_ar_pul_np * V_ar_pul_np)) *
                                  pow((dcb_c_o2_dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                              dcb_o2_dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) -
                                          dcb_o2_dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                              dcb_c_o2_dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np)),
                                      -1.) *
                                  (dcb_c_o2_dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                          (q_vout_r_np * (cb_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) -
                                                             cb_o2(ppCO2_v_r_np, ppO2_v_r_np))) -
                                      dcb_o2_dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                          (q_vout_r_np * (cb_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) -
                                                             cb_c_o2(ppCO2_v_r_np, ppO2_v_r_np)))));
    // w.r.t. upstream flux
    wkstiff(49, 29) =
        theta * ((1. / V_ar_pul_np) *
                        pow((dcb_c_o2_dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                    dcb_o2_dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) -
                                dcb_o2_dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                    dcb_c_o2_dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np)),
                            -1.) *
                        (dcb_c_o2_dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                (1.0 * (cb_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) -
                                           cb_o2(ppCO2_v_r_np, ppO2_v_r_np))) -
                            dcb_o2_dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                (1.0 * (cb_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) -
                                           cb_c_o2(ppCO2_v_r_np, ppO2_v_r_np)))) +
                    (-c_ar_pul_ * z_ar_pul_) * (-1. / (V_ar_pul_np * V_ar_pul_np)) *
                        pow((dcb_c_o2_dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                    dcb_o2_dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) -
                                dcb_o2_dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                    dcb_c_o2_dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np)),
                            -1.) *
                        (dcb_c_o2_dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                (q_vout_r_np * (cb_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) -
                                                   cb_o2(ppCO2_v_r_np, ppO2_v_r_np))) -
                            dcb_o2_dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                (q_vout_r_np * (cb_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) -
                                                   cb_c_o2(ppCO2_v_r_np, ppO2_v_r_np)))));
    // w.r.t. upstream ppCO2
    wkstiff(49, 46) =
        theta * (-(1. / V_ar_pul_np) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                dcb_o2_dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) -
                            dcb_o2_dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                dcb_c_o2_dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np)),
                        -1.) *
                    (dcb_c_o2_dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) * q_vout_r_np *
                            dcb_o2_dpp_c_o2(ppCO2_v_r_np, ppO2_v_r_np) -
                        dcb_o2_dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) * q_vout_r_np *
                            dcb_c_o2_dpp_c_o2(ppCO2_v_r_np, ppO2_v_r_np)));
    // w.r.t. upstream ppO2
    wkstiff(49, 47) =
        theta * (-(1. / V_ar_pul_np) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                dcb_o2_dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) -
                            dcb_o2_dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                dcb_c_o2_dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np)),
                        -1.) *
                    (dcb_c_o2_dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) * q_vout_r_np *
                            dcb_o2_dpp_o2(ppCO2_v_r_np, ppO2_v_r_np) -
                        dcb_o2_dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) * q_vout_r_np *
                            dcb_c_o2_dpp_o2(ppCO2_v_r_np, ppO2_v_r_np)));
    // w.r.t. ppCO2
    wkstiff(49, 48) =
        theta * (-(1. / V_ar_pul_np) *
                        pow((dcb_c_o2_dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                    dcb_o2_dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) -
                                dcb_o2_dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                    dcb_c_o2_dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np)),
                            -2.) *
                        (d2cb_c_o2_dpp_c_o22(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                dcb_o2_dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) +
                            dcb_c_o2_dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                d2cb_o2_dpp_o2dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) -
                            d2cb_o2_dpp_c_o22(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                dcb_c_o2_dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) -
                            dcb_o2_dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np)) *
                        (dcb_c_o2_dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                (q_vout_r_np * (cb_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) -
                                                   cb_o2(ppCO2_v_r_np, ppO2_v_r_np))) -
                            dcb_o2_dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                (q_vout_r_np * (cb_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) -
                                                   cb_c_o2(ppCO2_v_r_np, ppO2_v_r_np)))) +
                    (1. / V_ar_pul_np) *
                        pow((dcb_c_o2_dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                    dcb_o2_dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) -
                                dcb_o2_dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                    dcb_c_o2_dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np)),
                            -1.) *
                        (d2cb_c_o2_dpp_c_o22(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                (q_vout_r_np * (cb_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) -
                                                   cb_o2(ppCO2_v_r_np, ppO2_v_r_np))) +
                            dcb_c_o2_dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) * q_vout_r_np *
                                dcb_o2_dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) -
                            d2cb_o2_dpp_c_o22(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                (q_vout_r_np * (cb_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) -
                                                   cb_c_o2(ppCO2_v_r_np, ppO2_v_r_np))) -
                            dcb_o2_dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) * q_vout_r_np *
                                dcb_c_o2_dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np)));
    // w.r.t. ppO2
    wkstiff(49, 49) =
        1. / ts_size +
        theta * (-(1. / V_ar_pul_np) *
                        pow((dcb_c_o2_dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                    dcb_o2_dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) -
                                dcb_o2_dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                    dcb_c_o2_dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np)),
                            -2.) *
                        (d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                dcb_o2_dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) +
                            dcb_c_o2_dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                d2cb_o2_dpp_o22(ppCO2_ar_pul_np, ppO2_ar_pul_np) -
                            d2cb_o2_dpp_o2dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                dcb_c_o2_dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) -
                            dcb_o2_dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                d2cb_c_o2_dpp_o22(ppCO2_ar_pul_np, ppO2_ar_pul_np)) *
                        (dcb_c_o2_dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                (q_vout_r_np * (cb_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) -
                                                   cb_o2(ppCO2_v_r_np, ppO2_v_r_np))) -
                            dcb_o2_dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                (q_vout_r_np * (cb_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) -
                                                   cb_c_o2(ppCO2_v_r_np, ppO2_v_r_np)))) +
                    (1. / V_ar_pul_np) *
                        pow((dcb_c_o2_dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                    dcb_o2_dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) -
                                dcb_o2_dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                    dcb_c_o2_dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np)),
                            -1.) *
                        (d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                (q_vout_r_np * (cb_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) -
                                                   cb_o2(ppCO2_v_r_np, ppO2_v_r_np))) +
                            dcb_c_o2_dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) * q_vout_r_np *
                                dcb_o2_dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) -
                            d2cb_o2_dpp_o2dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) *
                                (q_vout_r_np * (cb_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) -
                                                   cb_c_o2(ppCO2_v_r_np, ppO2_v_r_np))) -
                            dcb_o2_dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) * q_vout_r_np *
                                dcb_c_o2_dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np)));


    //////// pulmonary capillaries CO2
    // w.r.t. mech. pressure
    wkstiff(50, 30) =
        theta * (c_cap_pul_ * (-1. / (V_cap_pul_np * V_cap_pul_np)) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                                dcb_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) -
                            dcb_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                                dcb_c_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np)),
                        -1.) *
                    (dcb_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                            (q_ar_pul_np * (cb_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) -
                                               cb_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np)) +
                                kappa_c_o2_ * (ppCO2_cap_pul_np -
                                                  fCO2_alv_np * (p_alv_np - p_vap_water_37_))) -
                        dcb_c_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                            (q_ar_pul_np * (cb_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) -
                                               cb_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np)) +
                                kappa_o2_ * (ppO2_cap_pul_np -
                                                fO2_alv_np * (p_alv_np - p_vap_water_37_)))));
    // w.r.t. upstream flux
    wkstiff(50, 31) = theta * ((1. / V_cap_pul_np) *
                                  pow((dcb_c_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                                              dcb_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) -
                                          dcb_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                                              dcb_c_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np)),
                                      -1.) *
                                  (dcb_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) * 1.0 *
                                          (cb_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) -
                                              cb_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np)) -
                                      dcb_c_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) * 1.0 *
                                          (cb_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) -
                                              cb_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np))));
    // w.r.t. alveolar pressure p_alv
    wkstiff(50, 36) =
        theta *
        ((1. / V_cap_pul_np) *
            pow((dcb_c_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                        dcb_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) -
                    dcb_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                        dcb_c_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np)),
                -1.) *
            (dcb_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) * (kappa_c_o2_ * (-fCO2_alv_np)) -
                dcb_c_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) * (kappa_o2_ * (-fO2_alv_np))));
    // w.r.t. alveolar CO2 fraction fCO2_alv
    wkstiff(50, 37) = theta * ((1. / V_cap_pul_np) *
                                  pow((dcb_c_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                                              dcb_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) -
                                          dcb_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                                              dcb_c_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np)),
                                      -1.) *
                                  (dcb_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                                      (kappa_c_o2_ * (-(p_alv_np - p_vap_water_37_)))));
    // w.r.t. alveolar O2 fraction fO2_alv
    wkstiff(50, 38) = theta * ((1. / V_cap_pul_np) *
                                  pow((dcb_c_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                                              dcb_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) -
                                          dcb_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                                              dcb_c_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np)),
                                      -1.) *
                                  (-dcb_c_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                                      (kappa_o2_ * (-(p_alv_np - p_vap_water_37_)))));
    // w.r.t. upstream ppCO2
    wkstiff(50, 48) =
        theta * (-(1. / V_cap_pul_np) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                                dcb_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) -
                            dcb_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                                dcb_c_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np)),
                        -1.) *
                    (dcb_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) * q_ar_pul_np *
                            dcb_c_o2_dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) -
                        dcb_c_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) * q_ar_pul_np *
                            dcb_o2_dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np)));
    // w.r.t. upstream ppO2
    wkstiff(50, 49) =
        theta * (-(1. / V_cap_pul_np) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                                dcb_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) -
                            dcb_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                                dcb_c_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np)),
                        -1.) *
                    (dcb_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) * q_ar_pul_np *
                            dcb_c_o2_dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) -
                        dcb_c_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) * q_ar_pul_np *
                            dcb_o2_dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np)));
    // w.r.t. ppCO2
    wkstiff(50, 50) =
        1. / ts_size +
        theta *
            (-(1. / V_cap_pul_np) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                                dcb_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) -
                            dcb_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                                dcb_c_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np)),
                        -2.) *
                    (d2cb_c_o2_dpp_c_o22(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                            dcb_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) +
                        dcb_c_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                            d2cb_o2_dpp_o2dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) -
                        d2cb_o2_dpp_c_o22(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                            dcb_c_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) -
                        dcb_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                            d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np)) *
                    (dcb_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                            (q_ar_pul_np * (cb_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) -
                                               cb_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np)) +
                                kappa_c_o2_ * (ppCO2_cap_pul_np -
                                                  fCO2_alv_np * (p_alv_np - p_vap_water_37_))) -
                        dcb_c_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                            (q_ar_pul_np * (cb_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) -
                                               cb_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np)) +
                                kappa_o2_ * (ppO2_cap_pul_np -
                                                fO2_alv_np * (p_alv_np - p_vap_water_37_)))) +
                (1. / V_cap_pul_np) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                                dcb_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) -
                            dcb_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                                dcb_c_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np)),
                        -1.) *
                    (d2cb_o2_dpp_o2dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                            (q_ar_pul_np * (cb_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) -
                                               cb_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np)) +
                                kappa_c_o2_ * (ppCO2_cap_pul_np -
                                                  fCO2_alv_np * (p_alv_np - p_vap_water_37_))) +
                        dcb_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                            (q_ar_pul_np * dcb_c_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) +
                                kappa_c_o2_) -
                        d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                            (q_ar_pul_np * (cb_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) -
                                               cb_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np)) +
                                kappa_o2_ *
                                    (ppO2_cap_pul_np - fO2_alv_np * (p_alv_np - p_vap_water_37_))) -
                        dcb_c_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) * q_ar_pul_np *
                            dcb_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np)));
    // w.r.t. ppO2
    wkstiff(50, 51) =
        theta *
        (-(1. / V_cap_pul_np) *
                pow((dcb_c_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                            dcb_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) -
                        dcb_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                            dcb_c_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np)),
                    -2.) *
                (d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                        dcb_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) +
                    dcb_c_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                        d2cb_o2_dpp_o22(ppCO2_cap_pul_np, ppO2_cap_pul_np) -
                    d2cb_o2_dpp_o2dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                        dcb_c_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) -
                    dcb_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                        d2cb_c_o2_dpp_o22(ppCO2_cap_pul_np, ppO2_cap_pul_np)) *
                (dcb_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                        (q_ar_pul_np * (cb_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) -
                                           cb_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np)) +
                            kappa_c_o2_ *
                                (ppCO2_cap_pul_np - fCO2_alv_np * (p_alv_np - p_vap_water_37_))) -
                    dcb_c_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                        (q_ar_pul_np * (cb_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) -
                                           cb_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np)) +
                            kappa_o2_ *
                                (ppO2_cap_pul_np - fO2_alv_np * (p_alv_np - p_vap_water_37_)))) +
            (1. / V_cap_pul_np) *
                pow((dcb_c_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                            dcb_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) -
                        dcb_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                            dcb_c_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np)),
                    -1.) *
                (d2cb_o2_dpp_o22(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                        (q_ar_pul_np * (cb_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) -
                                           cb_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np)) +
                            kappa_c_o2_ *
                                (ppCO2_cap_pul_np - fCO2_alv_np * (p_alv_np - p_vap_water_37_))) +
                    dcb_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) * q_ar_pul_np *
                        dcb_c_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) -
                    d2cb_c_o2_dpp_o22(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                        (q_ar_pul_np * (cb_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) -
                                           cb_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np)) +
                            kappa_o2_ *
                                (ppO2_cap_pul_np - fO2_alv_np * (p_alv_np - p_vap_water_37_))) -
                    dcb_c_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                        (q_ar_pul_np * dcb_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) +
                            kappa_o2_)));

    //////// pulmonary capillaries O2
    // w.r.t. mech. pressure
    wkstiff(51, 30) =
        theta *
        (c_cap_pul_ * (-1. / (V_cap_pul_np * V_cap_pul_np)) *
            pow((dcb_c_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                        dcb_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) -
                    dcb_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                        dcb_c_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np)),
                -1.) *
            (dcb_c_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                    (q_ar_pul_np * (cb_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) -
                                       cb_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np)) +
                        kappa_o2_ * (ppO2_cap_pul_np - fO2_alv_np * (p_alv_np - p_vap_water_37_))) -
                dcb_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                    (q_ar_pul_np * (cb_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) -
                                       cb_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np)) +
                        kappa_c_o2_ *
                            (ppCO2_cap_pul_np - fCO2_alv_np * (p_alv_np - p_vap_water_37_)))));
    // w.r.t. upstream flux
    wkstiff(51, 31) = theta * ((1. / V_cap_pul_np) *
                                  pow((dcb_c_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                                              dcb_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) -
                                          dcb_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                                              dcb_c_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np)),
                                      -1.) *
                                  (dcb_c_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) * 1.0 *
                                          (cb_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) -
                                              cb_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np)) -
                                      dcb_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) * 1.0 *
                                          (cb_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) -
                                              cb_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np))));
    // w.r.t. alveolar pressure p_alv
    wkstiff(51, 36) =
        theta *
        ((1. / V_cap_pul_np) *
            pow((dcb_c_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                        dcb_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) -
                    dcb_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                        dcb_c_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np)),
                -1.) *
            (dcb_c_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) * (kappa_o2_ * (-fO2_alv_np)) -
                dcb_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                    (kappa_c_o2_ * (-fCO2_alv_np))));
    // w.r.t. alveolar CO2 fraction fCO2_alv
    wkstiff(51, 37) = theta * ((1. / V_cap_pul_np) *
                                  pow((dcb_c_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                                              dcb_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) -
                                          dcb_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                                              dcb_c_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np)),
                                      -1.) *
                                  (-dcb_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                                      (kappa_c_o2_ * (-(p_alv_np - p_vap_water_37_)))));
    // w.r.t. alveolar O2 fraction fO2_alv
    wkstiff(51, 38) = theta * ((1. / V_cap_pul_np) *
                                  pow((dcb_c_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                                              dcb_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) -
                                          dcb_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                                              dcb_c_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np)),
                                      -1.) *
                                  (dcb_c_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                                      (kappa_o2_ * (-(p_alv_np - p_vap_water_37_)))));
    // w.r.t. upstream ppCO2
    wkstiff(51, 48) =
        theta * (-(1. / V_cap_pul_np) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                                dcb_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) -
                            dcb_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                                dcb_c_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np)),
                        -1.) *
                    (dcb_c_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) * q_ar_pul_np *
                            dcb_o2_dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) -
                        dcb_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) * q_ar_pul_np *
                            dcb_c_o2_dpp_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np)));
    // w.r.t. upstream ppO2
    wkstiff(51, 49) =
        theta * (-(1. / V_cap_pul_np) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                                dcb_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) -
                            dcb_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                                dcb_c_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np)),
                        -1.) *
                    (dcb_c_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) * q_ar_pul_np *
                            dcb_o2_dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np) -
                        dcb_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) * q_ar_pul_np *
                            dcb_c_o2_dpp_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np)));
    // w.r.t. ppCO2
    wkstiff(51, 50) =
        theta *
        (-(1. / V_cap_pul_np) *
                pow((dcb_c_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                            dcb_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) -
                        dcb_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                            dcb_c_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np)),
                    -2.) *
                (d2cb_c_o2_dpp_c_o22(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                        dcb_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) +
                    dcb_c_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                        d2cb_o2_dpp_o2dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) -
                    d2cb_o2_dpp_c_o22(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                        dcb_c_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) -
                    dcb_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                        d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np)) *
                (dcb_c_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                        (q_ar_pul_np * (cb_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) -
                                           cb_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np)) +
                            kappa_o2_ *
                                (ppO2_cap_pul_np - fO2_alv_np * (p_alv_np - p_vap_water_37_))) -
                    dcb_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                        (q_ar_pul_np * (cb_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) -
                                           cb_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np)) +
                            kappa_c_o2_ *
                                (ppCO2_cap_pul_np - fCO2_alv_np * (p_alv_np - p_vap_water_37_)))) +
            (1. / V_cap_pul_np) *
                pow((dcb_c_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                            dcb_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) -
                        dcb_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                            dcb_c_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np)),
                    -1.) *
                (d2cb_c_o2_dpp_c_o22(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                        (q_ar_pul_np * (cb_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) -
                                           cb_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np)) +
                            kappa_o2_ *
                                (ppO2_cap_pul_np - fO2_alv_np * (p_alv_np - p_vap_water_37_))) +
                    dcb_c_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) * q_ar_pul_np *
                        dcb_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) -
                    d2cb_o2_dpp_c_o22(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                        (q_ar_pul_np * (cb_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) -
                                           cb_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np)) +
                            kappa_c_o2_ *
                                (ppCO2_cap_pul_np - fCO2_alv_np * (p_alv_np - p_vap_water_37_))) -
                    dcb_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                        (q_ar_pul_np * dcb_c_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) +
                            kappa_c_o2_)));
    // w.r.t. ppO2
    wkstiff(51, 51) =
        1. / ts_size +
        theta *
            (-(1. / V_cap_pul_np) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                                dcb_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) -
                            dcb_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                                dcb_c_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np)),
                        -2.) *
                    (d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                            dcb_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) +
                        dcb_c_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                            d2cb_o2_dpp_o22(ppCO2_cap_pul_np, ppO2_cap_pul_np) -
                        d2cb_o2_dpp_o2dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                            dcb_c_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) -
                        dcb_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                            d2cb_c_o2_dpp_o22(ppCO2_cap_pul_np, ppO2_cap_pul_np)) *
                    (dcb_c_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                            (q_ar_pul_np * (cb_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) -
                                               cb_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np)) +
                                kappa_o2_ *
                                    (ppO2_cap_pul_np - fO2_alv_np * (p_alv_np - p_vap_water_37_))) -
                        dcb_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                            (q_ar_pul_np * (cb_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) -
                                               cb_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np)) +
                                kappa_c_o2_ * (ppCO2_cap_pul_np -
                                                  fCO2_alv_np * (p_alv_np - p_vap_water_37_)))) +
                (1. / V_cap_pul_np) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                                dcb_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) -
                            dcb_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                                dcb_c_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np)),
                        -1.) *
                    (d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                            (q_ar_pul_np * (cb_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) -
                                               cb_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np)) +
                                kappa_o2_ *
                                    (ppO2_cap_pul_np - fO2_alv_np * (p_alv_np - p_vap_water_37_))) +
                        dcb_c_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                            (q_ar_pul_np * dcb_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) +
                                kappa_o2_) -
                        d2cb_o2_dpp_o2dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) *
                            (q_ar_pul_np * (cb_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) -
                                               cb_c_o2(ppCO2_ar_pul_np, ppO2_ar_pul_np)) +
                                kappa_c_o2_ * (ppCO2_cap_pul_np -
                                                  fCO2_alv_np * (p_alv_np - p_vap_water_37_))) -
                        dcb_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) * q_ar_pul_np *
                            dcb_c_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np)));


    //////// pulmonary veins CO2
    // w.r.t. mech. pressure
    wkstiff(52, 30) =
        theta * (c_ven_pul_ * (-1. / (V_ven_pul_np * V_ven_pul_np)) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) *
                                dcb_o2_dpp_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) -
                            dcb_o2_dpp_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) *
                                dcb_c_o2_dpp_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np)),
                        -1.) *
                    (dcb_o2_dpp_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) *
                            (q_cap_pul_np * (cb_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) -
                                                cb_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np))) -
                        dcb_c_o2_dpp_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) *
                            (q_cap_pul_np * (cb_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) -
                                                cb_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np)))));
    // w.r.t. upstream flux
    wkstiff(52, 31) = theta * ((1. / V_ven_pul_np) *
                                  pow((dcb_c_o2_dpp_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) *
                                              dcb_o2_dpp_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) -
                                          dcb_o2_dpp_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) *
                                              dcb_c_o2_dpp_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np)),
                                      -1.) *
                                  (dcb_o2_dpp_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) *
                                          (1.0 * (cb_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) -
                                                     cb_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np))) -
                                      dcb_c_o2_dpp_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) *
                                          (1.0 * (cb_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) -
                                                     cb_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np)))));
    // w.r.t. upstream ppCO2
    wkstiff(52, 50) =
        theta * (-(1. / V_ven_pul_np) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) *
                                dcb_o2_dpp_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) -
                            dcb_o2_dpp_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) *
                                dcb_c_o2_dpp_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np)),
                        -1.) *
                    (dcb_o2_dpp_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) * q_cap_pul_np *
                            dcb_c_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) -
                        dcb_c_o2_dpp_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) * q_cap_pul_np *
                            dcb_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np)));
    // w.r.t. upstream ppO2
    wkstiff(52, 51) =
        theta * (-(1. / V_ven_pul_np) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) *
                                dcb_o2_dpp_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) -
                            dcb_o2_dpp_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) *
                                dcb_c_o2_dpp_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np)),
                        -1.) *
                    (dcb_o2_dpp_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) * q_cap_pul_np *
                            dcb_c_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) -
                        dcb_c_o2_dpp_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) * q_cap_pul_np *
                            dcb_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np)));
    // w.r.t. ppCO2
    wkstiff(52, 52) =
        1. / ts_size +
        theta * (-(1. / V_ven_pul_np) *
                        pow((dcb_c_o2_dpp_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) *
                                    dcb_o2_dpp_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) -
                                dcb_o2_dpp_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) *
                                    dcb_c_o2_dpp_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np)),
                            -2.) *
                        (d2cb_c_o2_dpp_c_o22(ppCO2_ven_pul_np, ppO2_ven_pul_np) *
                                dcb_o2_dpp_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) +
                            dcb_c_o2_dpp_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) *
                                d2cb_o2_dpp_o2dpp_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) -
                            d2cb_o2_dpp_c_o22(ppCO2_ven_pul_np, ppO2_ven_pul_np) *
                                dcb_c_o2_dpp_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) -
                            dcb_o2_dpp_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) *
                                d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np)) *
                        (dcb_o2_dpp_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) *
                                (q_cap_pul_np * (cb_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) -
                                                    cb_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np))) -
                            dcb_c_o2_dpp_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) *
                                (q_cap_pul_np * (cb_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) -
                                                    cb_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np)))) +
                    (1. / V_ven_pul_np) *
                        pow((dcb_c_o2_dpp_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) *
                                    dcb_o2_dpp_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) -
                                dcb_o2_dpp_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) *
                                    dcb_c_o2_dpp_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np)),
                            -1.) *
                        (d2cb_o2_dpp_o2dpp_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) *
                                (q_cap_pul_np * (cb_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) -
                                                    cb_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np))) +
                            dcb_o2_dpp_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) * q_cap_pul_np *
                                dcb_c_o2_dpp_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) -
                            d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) *
                                (q_cap_pul_np * (cb_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) -
                                                    cb_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np))) -
                            dcb_c_o2_dpp_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) * q_cap_pul_np *
                                dcb_o2_dpp_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np)));
    // w.r.t. ppO2
    wkstiff(52, 53) =
        theta * (-(1. / V_ven_pul_np) *
                        pow((dcb_c_o2_dpp_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) *
                                    dcb_o2_dpp_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) -
                                dcb_o2_dpp_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) *
                                    dcb_c_o2_dpp_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np)),
                            -2.) *
                        (d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) *
                                dcb_o2_dpp_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) +
                            dcb_c_o2_dpp_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) *
                                d2cb_o2_dpp_o22(ppCO2_ven_pul_np, ppO2_ven_pul_np) -
                            d2cb_o2_dpp_o2dpp_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) *
                                dcb_c_o2_dpp_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) -
                            dcb_o2_dpp_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) *
                                d2cb_c_o2_dpp_o22(ppCO2_ven_pul_np, ppO2_ven_pul_np)) *
                        (dcb_o2_dpp_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) *
                                (q_cap_pul_np * (cb_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) -
                                                    cb_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np))) -
                            dcb_c_o2_dpp_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) *
                                (q_cap_pul_np * (cb_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) -
                                                    cb_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np)))) +
                    (1. / V_ven_pul_np) *
                        pow((dcb_c_o2_dpp_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) *
                                    dcb_o2_dpp_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) -
                                dcb_o2_dpp_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) *
                                    dcb_c_o2_dpp_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np)),
                            -1.) *
                        (d2cb_o2_dpp_o22(ppCO2_ven_pul_np, ppO2_ven_pul_np) *
                                (q_cap_pul_np * (cb_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) -
                                                    cb_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np))) +
                            dcb_o2_dpp_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) * q_cap_pul_np *
                                dcb_c_o2_dpp_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) -
                            d2cb_c_o2_dpp_o22(ppCO2_ven_pul_np, ppO2_ven_pul_np) *
                                (q_cap_pul_np * (cb_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) -
                                                    cb_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np))) -
                            dcb_c_o2_dpp_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) * q_cap_pul_np *
                                dcb_o2_dpp_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np)));

    //////// pulmonary veins O2
    // w.r.t. mech. pressure
    wkstiff(53, 30) =
        theta * (c_ven_pul_ * (-1. / (V_ven_pul_np * V_ven_pul_np)) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) *
                                dcb_o2_dpp_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) -
                            dcb_o2_dpp_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) *
                                dcb_c_o2_dpp_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np)),
                        -1.) *
                    (dcb_c_o2_dpp_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) *
                            (q_cap_pul_np * (cb_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) -
                                                cb_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np))) -
                        dcb_o2_dpp_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) *
                            (q_cap_pul_np * (cb_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) -
                                                cb_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np)))));
    // w.r.t. upstream flux
    wkstiff(53, 31) = theta * ((1. / V_ven_pul_np) *
                                  pow((dcb_c_o2_dpp_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) *
                                              dcb_o2_dpp_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) -
                                          dcb_o2_dpp_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) *
                                              dcb_c_o2_dpp_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np)),
                                      -1.) *
                                  (dcb_c_o2_dpp_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) *
                                          (1.0 * (cb_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) -
                                                     cb_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np))) -
                                      dcb_o2_dpp_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) *
                                          (1.0 * (cb_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) -
                                                     cb_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np)))));
    // w.r.t. upstream ppCO2
    wkstiff(53, 50) =
        theta * (-(1. / V_ven_pul_np) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) *
                                dcb_o2_dpp_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) -
                            dcb_o2_dpp_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) *
                                dcb_c_o2_dpp_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np)),
                        -1.) *
                    (dcb_c_o2_dpp_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) * q_cap_pul_np *
                            dcb_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) -
                        dcb_o2_dpp_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) * q_cap_pul_np *
                            dcb_c_o2_dpp_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np)));
    // w.r.t. upstream ppO2
    wkstiff(53, 51) =
        theta * (-(1. / V_ven_pul_np) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) *
                                dcb_o2_dpp_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) -
                            dcb_o2_dpp_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) *
                                dcb_c_o2_dpp_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np)),
                        -1.) *
                    (dcb_c_o2_dpp_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) * q_cap_pul_np *
                            dcb_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np) -
                        dcb_o2_dpp_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) * q_cap_pul_np *
                            dcb_c_o2_dpp_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np)));
    // w.r.t. ppCO2
    wkstiff(53, 52) =
        theta * (-(1. / V_ven_pul_np) *
                        pow((dcb_c_o2_dpp_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) *
                                    dcb_o2_dpp_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) -
                                dcb_o2_dpp_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) *
                                    dcb_c_o2_dpp_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np)),
                            -2.) *
                        (d2cb_c_o2_dpp_c_o22(ppCO2_ven_pul_np, ppO2_ven_pul_np) *
                                dcb_o2_dpp_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) +
                            dcb_c_o2_dpp_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) *
                                d2cb_o2_dpp_o2dpp_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) -
                            d2cb_o2_dpp_c_o22(ppCO2_ven_pul_np, ppO2_ven_pul_np) *
                                dcb_c_o2_dpp_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) -
                            dcb_o2_dpp_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) *
                                d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np)) *
                        (dcb_c_o2_dpp_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) *
                                (q_cap_pul_np * (cb_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) -
                                                    cb_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np))) -
                            dcb_o2_dpp_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) *
                                (q_cap_pul_np * (cb_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) -
                                                    cb_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np)))) +
                    (1. / V_ven_pul_np) *
                        pow((dcb_c_o2_dpp_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) *
                                    dcb_o2_dpp_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) -
                                dcb_o2_dpp_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) *
                                    dcb_c_o2_dpp_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np)),
                            -1.) *
                        (d2cb_c_o2_dpp_c_o22(ppCO2_ven_pul_np, ppO2_ven_pul_np) *
                                (q_cap_pul_np * (cb_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) -
                                                    cb_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np))) +
                            dcb_c_o2_dpp_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) * q_cap_pul_np *
                                dcb_o2_dpp_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) -
                            d2cb_o2_dpp_c_o22(ppCO2_ven_pul_np, ppO2_ven_pul_np) *
                                (q_cap_pul_np * (cb_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) -
                                                    cb_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np))) -
                            dcb_o2_dpp_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) * q_cap_pul_np *
                                dcb_c_o2_dpp_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np)));
    // w.r.t. ppO2
    wkstiff(53, 53) =
        1. / ts_size +
        theta * (-(1. / V_ven_pul_np) *
                        pow((dcb_c_o2_dpp_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) *
                                    dcb_o2_dpp_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) -
                                dcb_o2_dpp_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) *
                                    dcb_c_o2_dpp_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np)),
                            -2.) *
                        (d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) *
                                dcb_o2_dpp_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) +
                            dcb_c_o2_dpp_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) *
                                d2cb_o2_dpp_o22(ppCO2_ven_pul_np, ppO2_ven_pul_np) -
                            d2cb_o2_dpp_o2dpp_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) *
                                dcb_c_o2_dpp_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) -
                            dcb_o2_dpp_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) *
                                d2cb_c_o2_dpp_o22(ppCO2_ven_pul_np, ppO2_ven_pul_np)) *
                        (dcb_c_o2_dpp_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) *
                                (q_cap_pul_np * (cb_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) -
                                                    cb_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np))) -
                            dcb_o2_dpp_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) *
                                (q_cap_pul_np * (cb_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) -
                                                    cb_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np)))) +
                    (1. / V_ven_pul_np) *
                        pow((dcb_c_o2_dpp_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) *
                                    dcb_o2_dpp_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) -
                                dcb_o2_dpp_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) *
                                    dcb_c_o2_dpp_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np)),
                            -1.) *
                        (d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) *
                                (q_cap_pul_np * (cb_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) -
                                                    cb_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np))) +
                            dcb_c_o2_dpp_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) * q_cap_pul_np *
                                dcb_o2_dpp_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) -
                            d2cb_o2_dpp_o2dpp_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) *
                                (q_cap_pul_np * (cb_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) -
                                                    cb_c_o2(ppCO2_cap_pul_np, ppO2_cap_pul_np))) -
                            dcb_o2_dpp_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) * q_cap_pul_np *
                                dcb_c_o2_dpp_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np)));


    //////// left atrium CO2
    // w.r.t. mech. pressure
    wkstiff(54, 32) =
        theta * (dV_at_l_dp * (-1. / (V_at_l_np * V_at_l_np)) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_at_l_np, ppO2_at_l_np) *
                                dcb_o2_dpp_o2(ppCO2_at_l_np, ppO2_at_l_np) -
                            dcb_o2_dpp_c_o2(ppCO2_at_l_np, ppO2_at_l_np) *
                                dcb_c_o2_dpp_o2(ppCO2_at_l_np, ppO2_at_l_np)),
                        -1.) *
                    (dcb_o2_dpp_o2(ppCO2_at_l_np, ppO2_at_l_np) *
                            (q_ven_pul_np * (cb_c_o2(ppCO2_at_l_np, ppO2_at_l_np) -
                                                cb_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np))) -
                        dcb_c_o2_dpp_o2(ppCO2_at_l_np, ppO2_at_l_np) *
                            (q_ven_pul_np * (cb_o2(ppCO2_at_l_np, ppO2_at_l_np) -
                                                cb_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np)))));
    // w.r.t. upstream flux
    wkstiff(54, 33) = theta * ((1. / V_at_l_np) *
                                  pow((dcb_c_o2_dpp_c_o2(ppCO2_at_l_np, ppO2_at_l_np) *
                                              dcb_o2_dpp_o2(ppCO2_at_l_np, ppO2_at_l_np) -
                                          dcb_o2_dpp_c_o2(ppCO2_at_l_np, ppO2_at_l_np) *
                                              dcb_c_o2_dpp_o2(ppCO2_at_l_np, ppO2_at_l_np)),
                                      -1.) *
                                  (dcb_o2_dpp_o2(ppCO2_at_l_np, ppO2_at_l_np) *
                                          (1.0 * (cb_c_o2(ppCO2_at_l_np, ppO2_at_l_np) -
                                                     cb_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np))) -
                                      dcb_c_o2_dpp_o2(ppCO2_at_l_np, ppO2_at_l_np) *
                                          (1.0 * (cb_o2(ppCO2_at_l_np, ppO2_at_l_np) -
                                                     cb_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np)))));
    // w.r.t. upstream ppCO2
    wkstiff(54, 52) = theta * (-(1. / V_at_l_np) *
                                  pow((dcb_c_o2_dpp_c_o2(ppCO2_at_l_np, ppO2_at_l_np) *
                                              dcb_o2_dpp_o2(ppCO2_at_l_np, ppO2_at_l_np) -
                                          dcb_o2_dpp_c_o2(ppCO2_at_l_np, ppO2_at_l_np) *
                                              dcb_c_o2_dpp_o2(ppCO2_at_l_np, ppO2_at_l_np)),
                                      -1.) *
                                  (dcb_o2_dpp_o2(ppCO2_at_l_np, ppO2_at_l_np) * q_ven_pul_np *
                                          dcb_c_o2_dpp_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) -
                                      dcb_c_o2_dpp_o2(ppCO2_at_l_np, ppO2_at_l_np) * q_ven_pul_np *
                                          dcb_o2_dpp_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np)));
    // w.r.t. upstream ppO2
    wkstiff(54, 53) = theta * (-(1. / V_at_l_np) *
                                  pow((dcb_c_o2_dpp_c_o2(ppCO2_at_l_np, ppO2_at_l_np) *
                                              dcb_o2_dpp_o2(ppCO2_at_l_np, ppO2_at_l_np) -
                                          dcb_o2_dpp_c_o2(ppCO2_at_l_np, ppO2_at_l_np) *
                                              dcb_c_o2_dpp_o2(ppCO2_at_l_np, ppO2_at_l_np)),
                                      -1.) *
                                  (dcb_o2_dpp_o2(ppCO2_at_l_np, ppO2_at_l_np) * q_ven_pul_np *
                                          dcb_c_o2_dpp_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) -
                                      dcb_c_o2_dpp_o2(ppCO2_at_l_np, ppO2_at_l_np) * q_ven_pul_np *
                                          dcb_o2_dpp_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np)));
    // w.r.t. ppCO2
    wkstiff(54, 54) =
        1. / ts_size +
        theta * (-(1. / V_at_l_np) *
                        pow((dcb_c_o2_dpp_c_o2(ppCO2_at_l_np, ppO2_at_l_np) *
                                    dcb_o2_dpp_o2(ppCO2_at_l_np, ppO2_at_l_np) -
                                dcb_o2_dpp_c_o2(ppCO2_at_l_np, ppO2_at_l_np) *
                                    dcb_c_o2_dpp_o2(ppCO2_at_l_np, ppO2_at_l_np)),
                            -2.) *
                        (d2cb_c_o2_dpp_c_o22(ppCO2_at_l_np, ppO2_at_l_np) *
                                dcb_o2_dpp_o2(ppCO2_at_l_np, ppO2_at_l_np) +
                            dcb_c_o2_dpp_c_o2(ppCO2_at_l_np, ppO2_at_l_np) *
                                d2cb_o2_dpp_o2dpp_c_o2(ppCO2_at_l_np, ppO2_at_l_np) -
                            d2cb_o2_dpp_c_o22(ppCO2_at_l_np, ppO2_at_l_np) *
                                dcb_c_o2_dpp_o2(ppCO2_at_l_np, ppO2_at_l_np) -
                            dcb_o2_dpp_c_o2(ppCO2_at_l_np, ppO2_at_l_np) *
                                d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_at_l_np, ppO2_at_l_np)) *
                        (dcb_o2_dpp_o2(ppCO2_at_l_np, ppO2_at_l_np) *
                                (q_ven_pul_np * (cb_c_o2(ppCO2_at_l_np, ppO2_at_l_np) -
                                                    cb_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np))) -
                            dcb_c_o2_dpp_o2(ppCO2_at_l_np, ppO2_at_l_np) *
                                (q_ven_pul_np * (cb_o2(ppCO2_at_l_np, ppO2_at_l_np) -
                                                    cb_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np)))) +
                    (1. / V_at_l_np) *
                        pow((dcb_c_o2_dpp_c_o2(ppCO2_at_l_np, ppO2_at_l_np) *
                                    dcb_o2_dpp_o2(ppCO2_at_l_np, ppO2_at_l_np) -
                                dcb_o2_dpp_c_o2(ppCO2_at_l_np, ppO2_at_l_np) *
                                    dcb_c_o2_dpp_o2(ppCO2_at_l_np, ppO2_at_l_np)),
                            -1.) *
                        (d2cb_o2_dpp_o2dpp_c_o2(ppCO2_at_l_np, ppO2_at_l_np) *
                                (q_ven_pul_np * (cb_c_o2(ppCO2_at_l_np, ppO2_at_l_np) -
                                                    cb_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np))) +
                            dcb_o2_dpp_o2(ppCO2_at_l_np, ppO2_at_l_np) * q_ven_pul_np *
                                dcb_c_o2_dpp_c_o2(ppCO2_at_l_np, ppO2_at_l_np) -
                            d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_at_l_np, ppO2_at_l_np) *
                                (q_ven_pul_np * (cb_o2(ppCO2_at_l_np, ppO2_at_l_np) -
                                                    cb_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np))) -
                            dcb_c_o2_dpp_o2(ppCO2_at_l_np, ppO2_at_l_np) * q_ven_pul_np *
                                dcb_o2_dpp_c_o2(ppCO2_at_l_np, ppO2_at_l_np)));
    // w.r.t. ppO2
    wkstiff(54, 55) =
        theta * (-(1. / V_at_l_np) *
                        pow((dcb_c_o2_dpp_c_o2(ppCO2_at_l_np, ppO2_at_l_np) *
                                    dcb_o2_dpp_o2(ppCO2_at_l_np, ppO2_at_l_np) -
                                dcb_o2_dpp_c_o2(ppCO2_at_l_np, ppO2_at_l_np) *
                                    dcb_c_o2_dpp_o2(ppCO2_at_l_np, ppO2_at_l_np)),
                            -2.) *
                        (d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_at_l_np, ppO2_at_l_np) *
                                dcb_o2_dpp_o2(ppCO2_at_l_np, ppO2_at_l_np) +
                            dcb_c_o2_dpp_c_o2(ppCO2_at_l_np, ppO2_at_l_np) *
                                d2cb_o2_dpp_o22(ppCO2_at_l_np, ppO2_at_l_np) -
                            d2cb_o2_dpp_o2dpp_c_o2(ppCO2_at_l_np, ppO2_at_l_np) *
                                dcb_c_o2_dpp_o2(ppCO2_at_l_np, ppO2_at_l_np) -
                            dcb_o2_dpp_c_o2(ppCO2_at_l_np, ppO2_at_l_np) *
                                d2cb_c_o2_dpp_o22(ppCO2_at_l_np, ppO2_at_l_np)) *
                        (dcb_o2_dpp_o2(ppCO2_at_l_np, ppO2_at_l_np) *
                                (q_ven_pul_np * (cb_c_o2(ppCO2_at_l_np, ppO2_at_l_np) -
                                                    cb_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np))) -
                            dcb_c_o2_dpp_o2(ppCO2_at_l_np, ppO2_at_l_np) *
                                (q_ven_pul_np * (cb_o2(ppCO2_at_l_np, ppO2_at_l_np) -
                                                    cb_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np)))) +
                    (1. / V_at_l_np) *
                        pow((dcb_c_o2_dpp_c_o2(ppCO2_at_l_np, ppO2_at_l_np) *
                                    dcb_o2_dpp_o2(ppCO2_at_l_np, ppO2_at_l_np) -
                                dcb_o2_dpp_c_o2(ppCO2_at_l_np, ppO2_at_l_np) *
                                    dcb_c_o2_dpp_o2(ppCO2_at_l_np, ppO2_at_l_np)),
                            -1.) *
                        (d2cb_o2_dpp_o22(ppCO2_at_l_np, ppO2_at_l_np) *
                                (q_ven_pul_np * (cb_c_o2(ppCO2_at_l_np, ppO2_at_l_np) -
                                                    cb_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np))) +
                            dcb_o2_dpp_o2(ppCO2_at_l_np, ppO2_at_l_np) * q_ven_pul_np *
                                dcb_c_o2_dpp_o2(ppCO2_at_l_np, ppO2_at_l_np) -
                            d2cb_c_o2_dpp_o22(ppCO2_at_l_np, ppO2_at_l_np) *
                                (q_ven_pul_np * (cb_o2(ppCO2_at_l_np, ppO2_at_l_np) -
                                                    cb_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np))) -
                            dcb_c_o2_dpp_o2(ppCO2_at_l_np, ppO2_at_l_np) * q_ven_pul_np *
                                dcb_o2_dpp_o2(ppCO2_at_l_np, ppO2_at_l_np)));

    //////// left atrium O2
    // w.r.t. mech. pressure
    wkstiff(55, 32) =
        theta * (dV_at_l_dp * (-1. / (V_at_l_np * V_at_l_np)) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_at_l_np, ppO2_at_l_np) *
                                dcb_o2_dpp_o2(ppCO2_at_l_np, ppO2_at_l_np) -
                            dcb_o2_dpp_c_o2(ppCO2_at_l_np, ppO2_at_l_np) *
                                dcb_c_o2_dpp_o2(ppCO2_at_l_np, ppO2_at_l_np)),
                        -1.) *
                    (dcb_c_o2_dpp_c_o2(ppCO2_at_l_np, ppO2_at_l_np) *
                            (q_ven_pul_np * (cb_o2(ppCO2_at_l_np, ppO2_at_l_np) -
                                                cb_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np))) -
                        dcb_o2_dpp_c_o2(ppCO2_at_l_np, ppO2_at_l_np) *
                            (q_ven_pul_np * (cb_c_o2(ppCO2_at_l_np, ppO2_at_l_np) -
                                                cb_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np)))));
    // w.r.t. upstream flux
    wkstiff(55, 33) = theta * ((1. / V_at_l_np) *
                                  pow((dcb_c_o2_dpp_c_o2(ppCO2_at_l_np, ppO2_at_l_np) *
                                              dcb_o2_dpp_o2(ppCO2_at_l_np, ppO2_at_l_np) -
                                          dcb_o2_dpp_c_o2(ppCO2_at_l_np, ppO2_at_l_np) *
                                              dcb_c_o2_dpp_o2(ppCO2_at_l_np, ppO2_at_l_np)),
                                      -1.) *
                                  (dcb_c_o2_dpp_c_o2(ppCO2_at_l_np, ppO2_at_l_np) *
                                          (1.0 * (cb_o2(ppCO2_at_l_np, ppO2_at_l_np) -
                                                     cb_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np))) -
                                      dcb_o2_dpp_c_o2(ppCO2_at_l_np, ppO2_at_l_np) *
                                          (1.0 * (cb_c_o2(ppCO2_at_l_np, ppO2_at_l_np) -
                                                     cb_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np)))));
    // w.r.t. upstream ppCO2
    wkstiff(55, 52) = theta * (-(1. / V_at_l_np) *
                                  pow((dcb_c_o2_dpp_c_o2(ppCO2_at_l_np, ppO2_at_l_np) *
                                              dcb_o2_dpp_o2(ppCO2_at_l_np, ppO2_at_l_np) -
                                          dcb_o2_dpp_c_o2(ppCO2_at_l_np, ppO2_at_l_np) *
                                              dcb_c_o2_dpp_o2(ppCO2_at_l_np, ppO2_at_l_np)),
                                      -1.) *
                                  (dcb_c_o2_dpp_c_o2(ppCO2_at_l_np, ppO2_at_l_np) * q_ven_pul_np *
                                          dcb_o2_dpp_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) -
                                      dcb_o2_dpp_c_o2(ppCO2_at_l_np, ppO2_at_l_np) * q_ven_pul_np *
                                          dcb_c_o2_dpp_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np)));
    // w.r.t. upstream ppO2
    wkstiff(55, 53) = theta * (-(1. / V_at_l_np) *
                                  pow((dcb_c_o2_dpp_c_o2(ppCO2_at_l_np, ppO2_at_l_np) *
                                              dcb_o2_dpp_o2(ppCO2_at_l_np, ppO2_at_l_np) -
                                          dcb_o2_dpp_c_o2(ppCO2_at_l_np, ppO2_at_l_np) *
                                              dcb_c_o2_dpp_o2(ppCO2_at_l_np, ppO2_at_l_np)),
                                      -1.) *
                                  (dcb_c_o2_dpp_c_o2(ppCO2_at_l_np, ppO2_at_l_np) * q_ven_pul_np *
                                          dcb_o2_dpp_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np) -
                                      dcb_o2_dpp_c_o2(ppCO2_at_l_np, ppO2_at_l_np) * q_ven_pul_np *
                                          dcb_c_o2_dpp_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np)));
    // w.r.t. ppCO2
    wkstiff(55, 54) =
        theta * (-(1. / V_at_l_np) *
                        pow((dcb_c_o2_dpp_c_o2(ppCO2_at_l_np, ppO2_at_l_np) *
                                    dcb_o2_dpp_o2(ppCO2_at_l_np, ppO2_at_l_np) -
                                dcb_o2_dpp_c_o2(ppCO2_at_l_np, ppO2_at_l_np) *
                                    dcb_c_o2_dpp_o2(ppCO2_at_l_np, ppO2_at_l_np)),
                            -2.) *
                        (d2cb_c_o2_dpp_c_o22(ppCO2_at_l_np, ppO2_at_l_np) *
                                dcb_o2_dpp_o2(ppCO2_at_l_np, ppO2_at_l_np) +
                            dcb_c_o2_dpp_c_o2(ppCO2_at_l_np, ppO2_at_l_np) *
                                d2cb_o2_dpp_o2dpp_c_o2(ppCO2_at_l_np, ppO2_at_l_np) -
                            d2cb_o2_dpp_c_o22(ppCO2_at_l_np, ppO2_at_l_np) *
                                dcb_c_o2_dpp_o2(ppCO2_at_l_np, ppO2_at_l_np) -
                            dcb_o2_dpp_c_o2(ppCO2_at_l_np, ppO2_at_l_np) *
                                d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_at_l_np, ppO2_at_l_np)) *
                        (dcb_c_o2_dpp_c_o2(ppCO2_at_l_np, ppO2_at_l_np) *
                                (q_ven_pul_np * (cb_o2(ppCO2_at_l_np, ppO2_at_l_np) -
                                                    cb_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np))) -
                            dcb_o2_dpp_c_o2(ppCO2_at_l_np, ppO2_at_l_np) *
                                (q_ven_pul_np * (cb_c_o2(ppCO2_at_l_np, ppO2_at_l_np) -
                                                    cb_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np)))) +
                    (1. / V_at_l_np) *
                        pow((dcb_c_o2_dpp_c_o2(ppCO2_at_l_np, ppO2_at_l_np) *
                                    dcb_o2_dpp_o2(ppCO2_at_l_np, ppO2_at_l_np) -
                                dcb_o2_dpp_c_o2(ppCO2_at_l_np, ppO2_at_l_np) *
                                    dcb_c_o2_dpp_o2(ppCO2_at_l_np, ppO2_at_l_np)),
                            -1.) *
                        (d2cb_c_o2_dpp_c_o22(ppCO2_at_l_np, ppO2_at_l_np) *
                                (q_ven_pul_np * (cb_o2(ppCO2_at_l_np, ppO2_at_l_np) -
                                                    cb_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np))) +
                            dcb_c_o2_dpp_c_o2(ppCO2_at_l_np, ppO2_at_l_np) * q_ven_pul_np *
                                dcb_o2_dpp_c_o2(ppCO2_at_l_np, ppO2_at_l_np) -
                            d2cb_o2_dpp_c_o22(ppCO2_at_l_np, ppO2_at_l_np) *
                                (q_ven_pul_np * (cb_c_o2(ppCO2_at_l_np, ppO2_at_l_np) -
                                                    cb_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np))) -
                            dcb_o2_dpp_c_o2(ppCO2_at_l_np, ppO2_at_l_np) * q_ven_pul_np *
                                dcb_c_o2_dpp_c_o2(ppCO2_at_l_np, ppO2_at_l_np)));
    // w.r.t. ppO2
    wkstiff(55, 55) =
        1. / ts_size +
        theta * (-(1. / V_at_l_np) *
                        pow((dcb_c_o2_dpp_c_o2(ppCO2_at_l_np, ppO2_at_l_np) *
                                    dcb_o2_dpp_o2(ppCO2_at_l_np, ppO2_at_l_np) -
                                dcb_o2_dpp_c_o2(ppCO2_at_l_np, ppO2_at_l_np) *
                                    dcb_c_o2_dpp_o2(ppCO2_at_l_np, ppO2_at_l_np)),
                            -2.) *
                        (d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_at_l_np, ppO2_at_l_np) *
                                dcb_o2_dpp_o2(ppCO2_at_l_np, ppO2_at_l_np) +
                            dcb_c_o2_dpp_c_o2(ppCO2_at_l_np, ppO2_at_l_np) *
                                d2cb_o2_dpp_o22(ppCO2_at_l_np, ppO2_at_l_np) -
                            d2cb_o2_dpp_o2dpp_c_o2(ppCO2_at_l_np, ppO2_at_l_np) *
                                dcb_c_o2_dpp_o2(ppCO2_at_l_np, ppO2_at_l_np) -
                            dcb_o2_dpp_c_o2(ppCO2_at_l_np, ppO2_at_l_np) *
                                d2cb_c_o2_dpp_o22(ppCO2_at_l_np, ppO2_at_l_np)) *
                        (dcb_c_o2_dpp_c_o2(ppCO2_at_l_np, ppO2_at_l_np) *
                                (q_ven_pul_np * (cb_o2(ppCO2_at_l_np, ppO2_at_l_np) -
                                                    cb_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np))) -
                            dcb_o2_dpp_c_o2(ppCO2_at_l_np, ppO2_at_l_np) *
                                (q_ven_pul_np * (cb_c_o2(ppCO2_at_l_np, ppO2_at_l_np) -
                                                    cb_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np)))) +
                    (1. / V_at_l_np) *
                        pow((dcb_c_o2_dpp_c_o2(ppCO2_at_l_np, ppO2_at_l_np) *
                                    dcb_o2_dpp_o2(ppCO2_at_l_np, ppO2_at_l_np) -
                                dcb_o2_dpp_c_o2(ppCO2_at_l_np, ppO2_at_l_np) *
                                    dcb_c_o2_dpp_o2(ppCO2_at_l_np, ppO2_at_l_np)),
                            -1.) *
                        (d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_at_l_np, ppO2_at_l_np) *
                                (q_ven_pul_np * (cb_o2(ppCO2_at_l_np, ppO2_at_l_np) -
                                                    cb_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np))) +
                            dcb_c_o2_dpp_c_o2(ppCO2_at_l_np, ppO2_at_l_np) * q_ven_pul_np *
                                dcb_o2_dpp_o2(ppCO2_at_l_np, ppO2_at_l_np) -
                            d2cb_o2_dpp_o2dpp_c_o2(ppCO2_at_l_np, ppO2_at_l_np) *
                                (q_ven_pul_np * (cb_c_o2(ppCO2_at_l_np, ppO2_at_l_np) -
                                                    cb_c_o2(ppCO2_ven_pul_np, ppO2_ven_pul_np))) -
                            dcb_o2_dpp_c_o2(ppCO2_at_l_np, ppO2_at_l_np) * q_ven_pul_np *
                                dcb_c_o2_dpp_o2(ppCO2_at_l_np, ppO2_at_l_np)));


    //////// left ventricle CO2
    // w.r.t. upstream flux
    wkstiff(56, 2) = theta * ((1. / V_v_l_np) *
                                 pow((dcb_c_o2_dpp_c_o2(ppCO2_v_l_np, ppO2_v_l_np) *
                                             dcb_o2_dpp_o2(ppCO2_v_l_np, ppO2_v_l_np) -
                                         dcb_o2_dpp_c_o2(ppCO2_v_l_np, ppO2_v_l_np) *
                                             dcb_c_o2_dpp_o2(ppCO2_v_l_np, ppO2_v_l_np)),
                                     -1.) *
                                 (dcb_o2_dpp_o2(ppCO2_v_l_np, ppO2_v_l_np) *
                                         (1.0 * (cb_c_o2(ppCO2_v_l_np, ppO2_v_l_np) -
                                                    cb_c_o2(ppCO2_at_l_np, ppO2_at_l_np))) -
                                     dcb_c_o2_dpp_o2(ppCO2_v_l_np, ppO2_v_l_np) *
                                         (1.0 * (cb_o2(ppCO2_v_l_np, ppO2_v_l_np) -
                                                    cb_o2(ppCO2_at_l_np, ppO2_at_l_np)))));
    // w.r.t. mech. pressure
    wkstiff(56, 3) = theta * (dV_v_l_dp * (-1. / (V_v_l_np * V_v_l_np)) *
                                 pow((dcb_c_o2_dpp_c_o2(ppCO2_v_l_np, ppO2_v_l_np) *
                                             dcb_o2_dpp_o2(ppCO2_v_l_np, ppO2_v_l_np) -
                                         dcb_o2_dpp_c_o2(ppCO2_v_l_np, ppO2_v_l_np) *
                                             dcb_c_o2_dpp_o2(ppCO2_v_l_np, ppO2_v_l_np)),
                                     -1.) *
                                 (dcb_o2_dpp_o2(ppCO2_v_l_np, ppO2_v_l_np) *
                                         (q_vin_l_np * (cb_c_o2(ppCO2_v_l_np, ppO2_v_l_np) -
                                                           cb_c_o2(ppCO2_at_l_np, ppO2_at_l_np))) -
                                     dcb_c_o2_dpp_o2(ppCO2_v_l_np, ppO2_v_l_np) *
                                         (q_vin_l_np * (cb_o2(ppCO2_v_l_np, ppO2_v_l_np) -
                                                           cb_o2(ppCO2_at_l_np, ppO2_at_l_np)))));
    // w.r.t. upstream ppCO2
    wkstiff(56, 54) = theta * (-(1. / V_v_l_np) *
                                  pow((dcb_c_o2_dpp_c_o2(ppCO2_v_l_np, ppO2_v_l_np) *
                                              dcb_o2_dpp_o2(ppCO2_v_l_np, ppO2_v_l_np) -
                                          dcb_o2_dpp_c_o2(ppCO2_v_l_np, ppO2_v_l_np) *
                                              dcb_c_o2_dpp_o2(ppCO2_v_l_np, ppO2_v_l_np)),
                                      -1.) *
                                  (dcb_o2_dpp_o2(ppCO2_v_l_np, ppO2_v_l_np) * q_vin_l_np *
                                          dcb_c_o2_dpp_c_o2(ppCO2_at_l_np, ppO2_at_l_np) -
                                      dcb_c_o2_dpp_o2(ppCO2_v_l_np, ppO2_v_l_np) * q_vin_l_np *
                                          dcb_o2_dpp_c_o2(ppCO2_at_l_np, ppO2_at_l_np)));
    // w.r.t. upstream ppO2
    wkstiff(56, 55) = theta * (-(1. / V_v_l_np) *
                                  pow((dcb_c_o2_dpp_c_o2(ppCO2_v_l_np, ppO2_v_l_np) *
                                              dcb_o2_dpp_o2(ppCO2_v_l_np, ppO2_v_l_np) -
                                          dcb_o2_dpp_c_o2(ppCO2_v_l_np, ppO2_v_l_np) *
                                              dcb_c_o2_dpp_o2(ppCO2_v_l_np, ppO2_v_l_np)),
                                      -1.) *
                                  (dcb_o2_dpp_o2(ppCO2_v_l_np, ppO2_v_l_np) * q_vin_l_np *
                                          dcb_c_o2_dpp_o2(ppCO2_at_l_np, ppO2_at_l_np) -
                                      dcb_c_o2_dpp_o2(ppCO2_v_l_np, ppO2_v_l_np) * q_vin_l_np *
                                          dcb_o2_dpp_o2(ppCO2_at_l_np, ppO2_at_l_np)));
    // w.r.t. ppCO2
    wkstiff(56, 56) =
        1. / ts_size +
        theta * (-(1. / V_v_l_np) *
                        pow((dcb_c_o2_dpp_c_o2(ppCO2_v_l_np, ppO2_v_l_np) *
                                    dcb_o2_dpp_o2(ppCO2_v_l_np, ppO2_v_l_np) -
                                dcb_o2_dpp_c_o2(ppCO2_v_l_np, ppO2_v_l_np) *
                                    dcb_c_o2_dpp_o2(ppCO2_v_l_np, ppO2_v_l_np)),
                            -2.) *
                        (d2cb_c_o2_dpp_c_o22(ppCO2_v_l_np, ppO2_v_l_np) *
                                dcb_o2_dpp_o2(ppCO2_v_l_np, ppO2_v_l_np) +
                            dcb_c_o2_dpp_c_o2(ppCO2_v_l_np, ppO2_v_l_np) *
                                d2cb_o2_dpp_o2dpp_c_o2(ppCO2_v_l_np, ppO2_v_l_np) -
                            d2cb_o2_dpp_c_o22(ppCO2_v_l_np, ppO2_v_l_np) *
                                dcb_c_o2_dpp_o2(ppCO2_v_l_np, ppO2_v_l_np) -
                            dcb_o2_dpp_c_o2(ppCO2_v_l_np, ppO2_v_l_np) *
                                d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_v_l_np, ppO2_v_l_np)) *
                        (dcb_o2_dpp_o2(ppCO2_v_l_np, ppO2_v_l_np) *
                                (q_vin_l_np * (cb_c_o2(ppCO2_v_l_np, ppO2_v_l_np) -
                                                  cb_c_o2(ppCO2_at_l_np, ppO2_at_l_np))) -
                            dcb_c_o2_dpp_o2(ppCO2_v_l_np, ppO2_v_l_np) *
                                (q_vin_l_np * (cb_o2(ppCO2_v_l_np, ppO2_v_l_np) -
                                                  cb_o2(ppCO2_at_l_np, ppO2_at_l_np)))) +
                    (1. / V_v_l_np) *
                        pow((dcb_c_o2_dpp_c_o2(ppCO2_v_l_np, ppO2_v_l_np) *
                                    dcb_o2_dpp_o2(ppCO2_v_l_np, ppO2_v_l_np) -
                                dcb_o2_dpp_c_o2(ppCO2_v_l_np, ppO2_v_l_np) *
                                    dcb_c_o2_dpp_o2(ppCO2_v_l_np, ppO2_v_l_np)),
                            -1.) *
                        (d2cb_o2_dpp_o2dpp_c_o2(ppCO2_v_l_np, ppO2_v_l_np) *
                                (q_vin_l_np * (cb_c_o2(ppCO2_v_l_np, ppO2_v_l_np) -
                                                  cb_c_o2(ppCO2_at_l_np, ppO2_at_l_np))) +
                            dcb_o2_dpp_o2(ppCO2_v_l_np, ppO2_v_l_np) * q_vin_l_np *
                                dcb_c_o2_dpp_c_o2(ppCO2_v_l_np, ppO2_v_l_np) -
                            d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_v_l_np, ppO2_v_l_np) *
                                (q_vin_l_np * (cb_o2(ppCO2_v_l_np, ppO2_v_l_np) -
                                                  cb_o2(ppCO2_at_l_np, ppO2_at_l_np))) -
                            dcb_c_o2_dpp_o2(ppCO2_v_l_np, ppO2_v_l_np) * q_vin_l_np *
                                dcb_o2_dpp_c_o2(ppCO2_v_l_np, ppO2_v_l_np)));
    // w.r.t. ppO2
    wkstiff(56, 57) =
        theta * (-(1. / V_v_l_np) *
                        pow((dcb_c_o2_dpp_c_o2(ppCO2_v_l_np, ppO2_v_l_np) *
                                    dcb_o2_dpp_o2(ppCO2_v_l_np, ppO2_v_l_np) -
                                dcb_o2_dpp_c_o2(ppCO2_v_l_np, ppO2_v_l_np) *
                                    dcb_c_o2_dpp_o2(ppCO2_v_l_np, ppO2_v_l_np)),
                            -2.) *
                        (d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_v_l_np, ppO2_v_l_np) *
                                dcb_o2_dpp_o2(ppCO2_v_l_np, ppO2_v_l_np) +
                            dcb_c_o2_dpp_c_o2(ppCO2_v_l_np, ppO2_v_l_np) *
                                d2cb_o2_dpp_o22(ppCO2_v_l_np, ppO2_v_l_np) -
                            d2cb_o2_dpp_o2dpp_c_o2(ppCO2_v_l_np, ppO2_v_l_np) *
                                dcb_c_o2_dpp_o2(ppCO2_v_l_np, ppO2_v_l_np) -
                            dcb_o2_dpp_c_o2(ppCO2_v_l_np, ppO2_v_l_np) *
                                d2cb_c_o2_dpp_o22(ppCO2_v_l_np, ppO2_v_l_np)) *
                        (dcb_o2_dpp_o2(ppCO2_v_l_np, ppO2_v_l_np) *
                                (q_vin_l_np * (cb_c_o2(ppCO2_v_l_np, ppO2_v_l_np) -
                                                  cb_c_o2(ppCO2_at_l_np, ppO2_at_l_np))) -
                            dcb_c_o2_dpp_o2(ppCO2_v_l_np, ppO2_v_l_np) *
                                (q_vin_l_np * (cb_o2(ppCO2_v_l_np, ppO2_v_l_np) -
                                                  cb_o2(ppCO2_at_l_np, ppO2_at_l_np)))) +
                    (1. / V_v_l_np) *
                        pow((dcb_c_o2_dpp_c_o2(ppCO2_v_l_np, ppO2_v_l_np) *
                                    dcb_o2_dpp_o2(ppCO2_v_l_np, ppO2_v_l_np) -
                                dcb_o2_dpp_c_o2(ppCO2_v_l_np, ppO2_v_l_np) *
                                    dcb_c_o2_dpp_o2(ppCO2_v_l_np, ppO2_v_l_np)),
                            -1.) *
                        (d2cb_o2_dpp_o22(ppCO2_v_l_np, ppO2_v_l_np) *
                                (q_vin_l_np * (cb_c_o2(ppCO2_v_l_np, ppO2_v_l_np) -
                                                  cb_c_o2(ppCO2_at_l_np, ppO2_at_l_np))) +
                            dcb_o2_dpp_o2(ppCO2_v_l_np, ppO2_v_l_np) * q_vin_l_np *
                                dcb_c_o2_dpp_o2(ppCO2_v_l_np, ppO2_v_l_np) -
                            d2cb_c_o2_dpp_o22(ppCO2_v_l_np, ppO2_v_l_np) *
                                (q_vin_l_np * (cb_o2(ppCO2_v_l_np, ppO2_v_l_np) -
                                                  cb_o2(ppCO2_at_l_np, ppO2_at_l_np))) -
                            dcb_c_o2_dpp_o2(ppCO2_v_l_np, ppO2_v_l_np) * q_vin_l_np *
                                dcb_o2_dpp_o2(ppCO2_v_l_np, ppO2_v_l_np)));

    //////// left ventricle O2
    // w.r.t. upstream flux
    wkstiff(57, 2) = theta * ((1. / V_v_l_np) *
                                 pow((dcb_c_o2_dpp_c_o2(ppCO2_v_l_np, ppO2_v_l_np) *
                                             dcb_o2_dpp_o2(ppCO2_v_l_np, ppO2_v_l_np) -
                                         dcb_o2_dpp_c_o2(ppCO2_v_l_np, ppO2_v_l_np) *
                                             dcb_c_o2_dpp_o2(ppCO2_v_l_np, ppO2_v_l_np)),
                                     -1.) *
                                 (dcb_c_o2_dpp_c_o2(ppCO2_v_l_np, ppO2_v_l_np) *
                                         (1.0 * (cb_o2(ppCO2_v_l_np, ppO2_v_l_np) -
                                                    cb_o2(ppCO2_at_l_np, ppO2_at_l_np))) -
                                     dcb_o2_dpp_c_o2(ppCO2_v_l_np, ppO2_v_l_np) *
                                         (1.0 * (cb_c_o2(ppCO2_v_l_np, ppO2_v_l_np) -
                                                    cb_c_o2(ppCO2_at_l_np, ppO2_at_l_np)))));
    // w.r.t. mech. pressure
    wkstiff(57, 3) = theta * (dV_v_l_dp * (-1. / (V_v_l_np * V_v_l_np)) *
                                 pow((dcb_c_o2_dpp_c_o2(ppCO2_v_l_np, ppO2_v_l_np) *
                                             dcb_o2_dpp_o2(ppCO2_v_l_np, ppO2_v_l_np) -
                                         dcb_o2_dpp_c_o2(ppCO2_v_l_np, ppO2_v_l_np) *
                                             dcb_c_o2_dpp_o2(ppCO2_v_l_np, ppO2_v_l_np)),
                                     -1.) *
                                 (dcb_c_o2_dpp_c_o2(ppCO2_v_l_np, ppO2_v_l_np) *
                                         (q_vin_l_np * (cb_o2(ppCO2_v_l_np, ppO2_v_l_np) -
                                                           cb_o2(ppCO2_at_l_np, ppO2_at_l_np))) -
                                     dcb_o2_dpp_c_o2(ppCO2_v_l_np, ppO2_v_l_np) *
                                         (q_vin_l_np * (cb_c_o2(ppCO2_v_l_np, ppO2_v_l_np) -
                                                           cb_c_o2(ppCO2_at_l_np, ppO2_at_l_np)))));
    // w.r.t. upstream ppCO2
    wkstiff(57, 54) = theta * (-(1. / V_v_l_np) *
                                  pow((dcb_c_o2_dpp_c_o2(ppCO2_v_l_np, ppO2_v_l_np) *
                                              dcb_o2_dpp_o2(ppCO2_v_l_np, ppO2_v_l_np) -
                                          dcb_o2_dpp_c_o2(ppCO2_v_l_np, ppO2_v_l_np) *
                                              dcb_c_o2_dpp_o2(ppCO2_v_l_np, ppO2_v_l_np)),
                                      -1.) *
                                  (dcb_c_o2_dpp_c_o2(ppCO2_v_l_np, ppO2_v_l_np) * q_vin_l_np *
                                          dcb_o2_dpp_c_o2(ppCO2_at_l_np, ppO2_at_l_np) -
                                      dcb_o2_dpp_c_o2(ppCO2_v_l_np, ppO2_v_l_np) * q_vin_l_np *
                                          dcb_c_o2_dpp_c_o2(ppCO2_at_l_np, ppO2_at_l_np)));
    // w.r.t. upstream ppO2
    wkstiff(57, 55) = theta * (-(1. / V_v_l_np) *
                                  pow((dcb_c_o2_dpp_c_o2(ppCO2_v_l_np, ppO2_v_l_np) *
                                              dcb_o2_dpp_o2(ppCO2_v_l_np, ppO2_v_l_np) -
                                          dcb_o2_dpp_c_o2(ppCO2_v_l_np, ppO2_v_l_np) *
                                              dcb_c_o2_dpp_o2(ppCO2_v_l_np, ppO2_v_l_np)),
                                      -1.) *
                                  (dcb_c_o2_dpp_c_o2(ppCO2_v_l_np, ppO2_v_l_np) * q_vin_l_np *
                                          dcb_o2_dpp_o2(ppCO2_at_l_np, ppO2_at_l_np) -
                                      dcb_o2_dpp_c_o2(ppCO2_v_l_np, ppO2_v_l_np) * q_vin_l_np *
                                          dcb_c_o2_dpp_o2(ppCO2_at_l_np, ppO2_at_l_np)));
    // w.r.t. ppCO2
    wkstiff(57, 56) =
        theta * (-(1. / V_v_l_np) *
                        pow((dcb_c_o2_dpp_c_o2(ppCO2_v_l_np, ppO2_v_l_np) *
                                    dcb_o2_dpp_o2(ppCO2_v_l_np, ppO2_v_l_np) -
                                dcb_o2_dpp_c_o2(ppCO2_v_l_np, ppO2_v_l_np) *
                                    dcb_c_o2_dpp_o2(ppCO2_v_l_np, ppO2_v_l_np)),
                            -2.) *
                        (d2cb_c_o2_dpp_c_o22(ppCO2_v_l_np, ppO2_v_l_np) *
                                dcb_o2_dpp_o2(ppCO2_v_l_np, ppO2_v_l_np) +
                            dcb_c_o2_dpp_c_o2(ppCO2_v_l_np, ppO2_v_l_np) *
                                d2cb_o2_dpp_o2dpp_c_o2(ppCO2_v_l_np, ppO2_v_l_np) -
                            d2cb_o2_dpp_c_o22(ppCO2_v_l_np, ppO2_v_l_np) *
                                dcb_c_o2_dpp_o2(ppCO2_v_l_np, ppO2_v_l_np) -
                            dcb_o2_dpp_c_o2(ppCO2_v_l_np, ppO2_v_l_np) *
                                d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_v_l_np, ppO2_v_l_np)) *
                        (dcb_c_o2_dpp_c_o2(ppCO2_v_l_np, ppO2_v_l_np) *
                                (q_vin_l_np * (cb_o2(ppCO2_v_l_np, ppO2_v_l_np) -
                                                  cb_o2(ppCO2_at_l_np, ppO2_at_l_np))) -
                            dcb_o2_dpp_c_o2(ppCO2_v_l_np, ppO2_v_l_np) *
                                (q_vin_l_np * (cb_c_o2(ppCO2_v_l_np, ppO2_v_l_np) -
                                                  cb_c_o2(ppCO2_at_l_np, ppO2_at_l_np)))) +
                    (1. / V_v_l_np) *
                        pow((dcb_c_o2_dpp_c_o2(ppCO2_v_l_np, ppO2_v_l_np) *
                                    dcb_o2_dpp_o2(ppCO2_v_l_np, ppO2_v_l_np) -
                                dcb_o2_dpp_c_o2(ppCO2_v_l_np, ppO2_v_l_np) *
                                    dcb_c_o2_dpp_o2(ppCO2_v_l_np, ppO2_v_l_np)),
                            -1.) *
                        (d2cb_c_o2_dpp_c_o22(ppCO2_v_l_np, ppO2_v_l_np) *
                                (q_vin_l_np * (cb_o2(ppCO2_v_l_np, ppO2_v_l_np) -
                                                  cb_o2(ppCO2_at_l_np, ppO2_at_l_np))) +
                            dcb_c_o2_dpp_c_o2(ppCO2_v_l_np, ppO2_v_l_np) * q_vin_l_np *
                                dcb_c_o2_dpp_o2(ppCO2_v_l_np, ppO2_v_l_np) -
                            d2cb_o2_dpp_c_o22(ppCO2_v_l_np, ppO2_v_l_np) *
                                (q_vin_l_np * (cb_c_o2(ppCO2_v_l_np, ppO2_v_l_np) -
                                                  cb_c_o2(ppCO2_at_l_np, ppO2_at_l_np))) -
                            dcb_o2_dpp_c_o2(ppCO2_v_l_np, ppO2_v_l_np) * q_vin_l_np *
                                dcb_c_o2_dpp_c_o2(ppCO2_v_l_np, ppO2_v_l_np)));
    // w.r.t. ppO2
    wkstiff(57, 57) =
        1. / ts_size +
        theta * (-(1. / V_v_l_np) *
                        pow((dcb_c_o2_dpp_c_o2(ppCO2_v_l_np, ppO2_v_l_np) *
                                    dcb_o2_dpp_o2(ppCO2_v_l_np, ppO2_v_l_np) -
                                dcb_o2_dpp_c_o2(ppCO2_v_l_np, ppO2_v_l_np) *
                                    dcb_c_o2_dpp_o2(ppCO2_v_l_np, ppO2_v_l_np)),
                            -2.) *
                        (d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_v_l_np, ppO2_v_l_np) *
                                dcb_o2_dpp_o2(ppCO2_v_l_np, ppO2_v_l_np) +
                            dcb_c_o2_dpp_c_o2(ppCO2_v_l_np, ppO2_v_l_np) *
                                d2cb_o2_dpp_o22(ppCO2_v_l_np, ppO2_v_l_np) -
                            d2cb_o2_dpp_o2dpp_c_o2(ppCO2_v_l_np, ppO2_v_l_np) *
                                dcb_c_o2_dpp_o2(ppCO2_v_l_np, ppO2_v_l_np) -
                            dcb_o2_dpp_c_o2(ppCO2_v_l_np, ppO2_v_l_np) *
                                d2cb_c_o2_dpp_o22(ppCO2_v_l_np, ppO2_v_l_np)) *
                        (dcb_c_o2_dpp_c_o2(ppCO2_v_l_np, ppO2_v_l_np) *
                                (q_vin_l_np * (cb_o2(ppCO2_v_l_np, ppO2_v_l_np) -
                                                  cb_o2(ppCO2_at_l_np, ppO2_at_l_np))) -
                            dcb_o2_dpp_c_o2(ppCO2_v_l_np, ppO2_v_l_np) *
                                (q_vin_l_np * (cb_c_o2(ppCO2_v_l_np, ppO2_v_l_np) -
                                                  cb_c_o2(ppCO2_at_l_np, ppO2_at_l_np)))) +
                    (1. / V_v_l_np) *
                        pow((dcb_c_o2_dpp_c_o2(ppCO2_v_l_np, ppO2_v_l_np) *
                                    dcb_o2_dpp_o2(ppCO2_v_l_np, ppO2_v_l_np) -
                                dcb_o2_dpp_c_o2(ppCO2_v_l_np, ppO2_v_l_np) *
                                    dcb_c_o2_dpp_o2(ppCO2_v_l_np, ppO2_v_l_np)),
                            -1.) *
                        (d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_v_l_np, ppO2_v_l_np) *
                                (q_vin_l_np * (cb_o2(ppCO2_v_l_np, ppO2_v_l_np) -
                                                  cb_o2(ppCO2_at_l_np, ppO2_at_l_np))) +
                            dcb_c_o2_dpp_c_o2(ppCO2_v_l_np, ppO2_v_l_np) * q_vin_l_np *
                                dcb_o2_dpp_o2(ppCO2_v_l_np, ppO2_v_l_np) -
                            d2cb_o2_dpp_o2dpp_c_o2(ppCO2_v_l_np, ppO2_v_l_np) *
                                (q_vin_l_np * (cb_c_o2(ppCO2_v_l_np, ppO2_v_l_np) -
                                                  cb_c_o2(ppCO2_at_l_np, ppO2_at_l_np))) -
                            dcb_o2_dpp_c_o2(ppCO2_v_l_np, ppO2_v_l_np) * q_vin_l_np *
                                dcb_c_o2_dpp_o2(ppCO2_v_l_np, ppO2_v_l_np)));


    //////// systemic arteries CO2
    // w.r.t. mech. pressure
    wkstiff(58, 4) = theta * (c_ar_sys_ * (-1. / (V_ar_sys_np * V_ar_sys_np)) *
                                 pow((dcb_c_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                             dcb_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) -
                                         dcb_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                             dcb_c_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)),
                                     -1.) *
                                 (dcb_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                         (q_vout_l_np * (cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) -
                                                            cb_c_o2(ppCO2_v_l_np, ppO2_v_l_np))) -
                                     dcb_c_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                         (q_vout_l_np * (cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) -
                                                            cb_o2(ppCO2_v_l_np, ppO2_v_l_np)))));
    // w.r.t. upstream flux
    wkstiff(58, 5) =
        theta * ((1. / V_ar_sys_np) *
                        pow((dcb_c_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                    dcb_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) -
                                dcb_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                    dcb_c_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)),
                            -1.) *
                        (dcb_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                (1.0 * (cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) -
                                           cb_c_o2(ppCO2_v_l_np, ppO2_v_l_np))) -
                            dcb_c_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                (1.0 * (cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) -
                                           cb_o2(ppCO2_v_l_np, ppO2_v_l_np)))) +
                    (-c_ar_sys_ * z_ar_sys_) * (-1. / (V_ar_sys_np * V_ar_sys_np)) *
                        pow((dcb_c_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                    dcb_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) -
                                dcb_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                    dcb_c_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)),
                            -1.) *
                        (dcb_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                (q_vout_l_np * (cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) -
                                                   cb_c_o2(ppCO2_v_l_np, ppO2_v_l_np))) -
                            dcb_c_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                (q_vout_l_np * (cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) -
                                                   cb_o2(ppCO2_v_l_np, ppO2_v_l_np)))));
    // w.r.t. upstream ppCO2
    wkstiff(58, 56) =
        theta * (-(1. / V_ar_sys_np) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                dcb_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) -
                            dcb_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)),
                        -1.) *
                    (dcb_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) * q_vout_l_np *
                            dcb_c_o2_dpp_c_o2(ppCO2_v_l_np, ppO2_v_l_np) -
                        dcb_c_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) * q_vout_l_np *
                            dcb_o2_dpp_c_o2(ppCO2_v_l_np, ppO2_v_l_np)));
    // w.r.t. upstream ppO2
    wkstiff(58, 57) = theta * (-(1. / V_ar_sys_np) *
                                  pow((dcb_c_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                              dcb_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) -
                                          dcb_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                              dcb_c_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)),
                                      -1.) *
                                  (dcb_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) * q_vout_l_np *
                                          dcb_c_o2_dpp_o2(ppCO2_v_l_np, ppO2_v_l_np) -
                                      dcb_c_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                          q_vout_l_np * dcb_o2_dpp_o2(ppCO2_v_l_np, ppO2_v_l_np)));
    // w.r.t. ppCO2
    wkstiff(58, 58) =
        1. / ts_size +
        theta * (-(1. / V_ar_sys_np) *
                        pow((dcb_c_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                    dcb_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) -
                                dcb_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                    dcb_c_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)),
                            -2.) *
                        (d2cb_c_o2_dpp_c_o22(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                dcb_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) +
                            dcb_c_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                d2cb_o2_dpp_o2dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) -
                            d2cb_o2_dpp_c_o22(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) -
                            dcb_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) *
                        (dcb_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                (q_vout_l_np * (cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) -
                                                   cb_c_o2(ppCO2_v_l_np, ppO2_v_l_np))) -
                            dcb_c_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                (q_vout_l_np * (cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) -
                                                   cb_o2(ppCO2_v_l_np, ppO2_v_l_np)))) +
                    (1. / V_ar_sys_np) *
                        pow((dcb_c_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                    dcb_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) -
                                dcb_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                    dcb_c_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)),
                            -1.) *
                        (d2cb_o2_dpp_o2dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                (q_vout_l_np * (cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) -
                                                   cb_c_o2(ppCO2_v_l_np, ppO2_v_l_np))) +
                            dcb_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) * q_vout_l_np *
                                dcb_c_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) -
                            d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                (q_vout_l_np * (cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) -
                                                   cb_o2(ppCO2_v_l_np, ppO2_v_l_np))) -
                            dcb_c_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) * q_vout_l_np *
                                dcb_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)));
    // w.r.t. ppO2
    wkstiff(58, 59) =
        theta * (-(1. / V_ar_sys_np) *
                        pow((dcb_c_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                    dcb_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) -
                                dcb_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                    dcb_c_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)),
                            -2.) *
                        (d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                dcb_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) +
                            dcb_c_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                d2cb_o2_dpp_o22(ppCO2_ar_sys_np, ppO2_ar_sys_np) -
                            d2cb_o2_dpp_o2dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) -
                            dcb_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                d2cb_c_o2_dpp_o22(ppCO2_ar_sys_np, ppO2_ar_sys_np)) *
                        (dcb_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                (q_vout_l_np * (cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) -
                                                   cb_c_o2(ppCO2_v_l_np, ppO2_v_l_np))) -
                            dcb_c_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                (q_vout_l_np * (cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) -
                                                   cb_o2(ppCO2_v_l_np, ppO2_v_l_np)))) +
                    (1. / V_ar_sys_np) *
                        pow((dcb_c_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                    dcb_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) -
                                dcb_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                    dcb_c_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)),
                            -1.) *
                        (d2cb_o2_dpp_o22(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                (q_vout_l_np * (cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) -
                                                   cb_c_o2(ppCO2_v_l_np, ppO2_v_l_np))) +
                            dcb_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) * q_vout_l_np *
                                dcb_c_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) -
                            d2cb_c_o2_dpp_o22(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                (q_vout_l_np * (cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) -
                                                   cb_o2(ppCO2_v_l_np, ppO2_v_l_np))) -
                            dcb_c_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) * q_vout_l_np *
                                dcb_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)));

    //////// systemic arteries O2
    // w.r.t. mech. pressure
    wkstiff(59, 4) = theta * (c_ar_sys_ * (-1. / (V_ar_sys_np * V_ar_sys_np)) *
                                 pow((dcb_c_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                             dcb_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) -
                                         dcb_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                             dcb_c_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)),
                                     -1.) *
                                 (dcb_c_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                         (q_vout_l_np * (cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) -
                                                            cb_o2(ppCO2_v_l_np, ppO2_v_l_np))) -
                                     dcb_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                         (q_vout_l_np * (cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) -
                                                            cb_c_o2(ppCO2_v_l_np, ppO2_v_l_np)))));
    // w.r.t. upstream flux
    wkstiff(59, 5) =
        theta * ((1. / V_ar_sys_np) *
                        pow((dcb_c_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                    dcb_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) -
                                dcb_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                    dcb_c_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)),
                            -1.) *
                        (dcb_c_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                (1.0 * (cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) -
                                           cb_o2(ppCO2_v_l_np, ppO2_v_l_np))) -
                            dcb_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                (1.0 * (cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) -
                                           cb_c_o2(ppCO2_v_l_np, ppO2_v_l_np)))) +
                    (-c_ar_sys_ * z_ar_sys_) * (-1. / (V_ar_sys_np * V_ar_sys_np)) *
                        pow((dcb_c_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                    dcb_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) -
                                dcb_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                    dcb_c_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)),
                            -1.) *
                        (dcb_c_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                (q_vout_l_np * (cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) -
                                                   cb_o2(ppCO2_v_l_np, ppO2_v_l_np))) -
                            dcb_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                (q_vout_l_np * (cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) -
                                                   cb_c_o2(ppCO2_v_l_np, ppO2_v_l_np)))));
    // w.r.t. upstream ppCO2
    wkstiff(59, 56) =
        theta * (-(1. / V_ar_sys_np) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                dcb_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) -
                            dcb_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)),
                        -1.) *
                    (dcb_c_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) * q_vout_l_np *
                            dcb_o2_dpp_c_o2(ppCO2_v_l_np, ppO2_v_l_np) -
                        dcb_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) * q_vout_l_np *
                            dcb_c_o2_dpp_c_o2(ppCO2_v_l_np, ppO2_v_l_np)));
    // w.r.t. upstream ppO2
    wkstiff(59, 57) =
        theta * (-(1. / V_ar_sys_np) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                dcb_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) -
                            dcb_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)),
                        -1.) *
                    (dcb_c_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) * q_vout_l_np *
                            dcb_o2_dpp_o2(ppCO2_v_l_np, ppO2_v_l_np) -
                        dcb_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) * q_vout_l_np *
                            dcb_c_o2_dpp_o2(ppCO2_v_l_np, ppO2_v_l_np)));
    // w.r.t. ppCO2
    wkstiff(59, 58) =
        theta * (-(1. / V_ar_sys_np) *
                        pow((dcb_c_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                    dcb_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) -
                                dcb_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                    dcb_c_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)),
                            -2.) *
                        (d2cb_c_o2_dpp_c_o22(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                dcb_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) +
                            dcb_c_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                d2cb_o2_dpp_o2dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) -
                            d2cb_o2_dpp_c_o22(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) -
                            dcb_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) *
                        (dcb_c_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                (q_vout_l_np * (cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) -
                                                   cb_o2(ppCO2_v_l_np, ppO2_v_l_np))) -
                            dcb_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                (q_vout_l_np * (cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) -
                                                   cb_c_o2(ppCO2_v_l_np, ppO2_v_l_np)))) +
                    (1. / V_ar_sys_np) *
                        pow((dcb_c_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                    dcb_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) -
                                dcb_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                    dcb_c_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)),
                            -1.) *
                        (d2cb_c_o2_dpp_c_o22(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                (q_vout_l_np * (cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) -
                                                   cb_o2(ppCO2_v_l_np, ppO2_v_l_np))) +
                            dcb_c_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) * q_vout_l_np *
                                dcb_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) -
                            d2cb_o2_dpp_c_o22(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                (q_vout_l_np * (cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) -
                                                   cb_c_o2(ppCO2_v_l_np, ppO2_v_l_np))) -
                            dcb_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) * q_vout_l_np *
                                dcb_c_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)));
    // w.r.t. ppO2
    wkstiff(59, 59) =
        1. / ts_size +
        theta * (-(1. / V_ar_sys_np) *
                        pow((dcb_c_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                    dcb_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) -
                                dcb_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                    dcb_c_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)),
                            -2.) *
                        (d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                dcb_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) +
                            dcb_c_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                d2cb_o2_dpp_o22(ppCO2_ar_sys_np, ppO2_ar_sys_np) -
                            d2cb_o2_dpp_o2dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) -
                            dcb_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                d2cb_c_o2_dpp_o22(ppCO2_ar_sys_np, ppO2_ar_sys_np)) *
                        (dcb_c_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                (q_vout_l_np * (cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) -
                                                   cb_o2(ppCO2_v_l_np, ppO2_v_l_np))) -
                            dcb_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                (q_vout_l_np * (cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) -
                                                   cb_c_o2(ppCO2_v_l_np, ppO2_v_l_np)))) +
                    (1. / V_ar_sys_np) *
                        pow((dcb_c_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                    dcb_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) -
                                dcb_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                    dcb_c_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)),
                            -1.) *
                        (d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                (q_vout_l_np * (cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) -
                                                   cb_o2(ppCO2_v_l_np, ppO2_v_l_np))) +
                            dcb_c_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) * q_vout_l_np *
                                dcb_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) -
                            d2cb_o2_dpp_o2dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) *
                                (q_vout_l_np * (cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) -
                                                   cb_c_o2(ppCO2_v_l_np, ppO2_v_l_np))) -
                            dcb_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) * q_vout_l_np *
                                dcb_c_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)));



    //////// systemic splanchnic arteries CO2
    // w.r.t. mech. pressure
    wkstiff(60, 6) =
        theta *
        (c_arspl_sys_ * (-1. / (V_arspl_sys_np * V_arspl_sys_np)) *
                pow(((dcb_c_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
                         (v_tissspl_ / V_arspl_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arspl_sys_np)) *
                            (dcb_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
                                (v_tissspl_ / V_arspl_sys_np) * dct_o2_dpp_o2(ppO2_arspl_sys_np)) -
                        dcb_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) *
                            dcb_c_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np)),
                    -1.) *
                ((dcb_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
                     (v_tissspl_ / V_arspl_sys_np) * dct_o2_dpp_o2(ppO2_arspl_sys_np)) *
                        (q_arspl_sys_in_np * (cb_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) -
                                                 cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                            m_c_o2_arspl_) -
                    dcb_c_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) *
                        (q_arspl_sys_in_np * (cb_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) -
                                                 cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) +
                            m_o2_arspl_ * ct_o2(ppO2_arspl_sys_np) /
                                (beta + ct_o2(ppO2_arspl_sys_np)))) +
            (1. / V_arspl_sys_np) *
                (pow(((dcb_c_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
                          (v_tissspl_ / V_arspl_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arspl_sys_np)) *
                             (dcb_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
                                 (v_tissspl_ / V_arspl_sys_np) * dct_o2_dpp_o2(ppO2_arspl_sys_np)) -
                         dcb_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) *
                             dcb_c_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np)),
                    -1.)) *
                (c_arspl_sys_ * (-v_tissspl_ / (V_arspl_sys_np * V_arspl_sys_np)) *
                    dct_o2_dpp_o2(ppO2_arspl_sys_np) *
                    (q_arspl_sys_in_np * (cb_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) -
                                             cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                        m_c_o2_arspl_) *
                    (-pow(((dcb_c_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
                               (v_tissspl_ / V_arspl_sys_np) *
                                   dct_c_o2_dpp_c_o2(ppCO2_arspl_sys_np)) *
                                  (dcb_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
                                      (v_tissspl_ / V_arspl_sys_np) *
                                          dct_o2_dpp_o2(ppO2_arspl_sys_np)) -
                              dcb_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) *
                                  dcb_c_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np)),
                         -2.) *
                            (dcb_c_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
                                (v_tissspl_ / V_arspl_sys_np) *
                                    dct_c_o2_dpp_c_o2(ppCO2_arspl_sys_np)) *
                            c_arspl_sys_ * (-v_tissspl_ / (V_arspl_sys_np * V_arspl_sys_np)) *
                            dct_o2_dpp_o2(ppO2_arspl_sys_np) +
                        c_arspl_sys_ * (-v_tissspl_ / (V_arspl_sys_np * V_arspl_sys_np)) *
                            dct_c_o2_dpp_c_o2(ppCO2_arspl_sys_np) *
                            (dcb_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
                                (v_tissspl_ / V_arspl_sys_np) * dct_o2_dpp_o2(ppO2_arspl_sys_np))) *
                    ((dcb_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
                         (v_tissspl_ / V_arspl_sys_np) * dct_o2_dpp_o2(ppO2_arspl_sys_np)) *
                            (q_arspl_sys_in_np * (cb_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) -
                                                     cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                                m_c_o2_arspl_) -
                        dcb_c_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) *
                            (q_arspl_sys_in_np * (cb_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) -
                                                     cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) +
                                m_o2_arspl_ * ct_o2(ppO2_arspl_sys_np) /
                                    (beta + ct_o2(ppO2_arspl_sys_np))))));
    // w.r.t. upstream flux
    wkstiff(60, 39) =
        theta *
        ((1. / V_arspl_sys_np) *
            pow(((dcb_c_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
                     (v_tissspl_ / V_arspl_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arspl_sys_np)) *
                        (dcb_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
                            (v_tissspl_ / V_arspl_sys_np) * dct_o2_dpp_o2(ppO2_arspl_sys_np)) -
                    dcb_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np)),
                -1.) *
            (dcb_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) * 1.0 *
                    (cb_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) -
                        cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                dcb_c_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) * 1.0 *
                    (cb_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) -
                        cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np))));
    // w.r.t. upstream ppCO2
    wkstiff(60, 58) =
        theta *
        (-(1. / V_arspl_sys_np) *
            pow(((dcb_c_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
                     (v_tissspl_ / V_arspl_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arspl_sys_np)) *
                        (dcb_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
                            (v_tissspl_ / V_arspl_sys_np) * dct_o2_dpp_o2(ppO2_arspl_sys_np)) -
                    dcb_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np)),
                -1.) *
            ((dcb_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
                 (v_tissspl_ / V_arspl_sys_np) * dct_o2_dpp_o2(ppO2_arspl_sys_np)) *
                    q_arspl_sys_in_np * dcb_c_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) -
                dcb_c_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) * q_arspl_sys_in_np *
                    dcb_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)));
    // w.r.t. upstream ppO2
    wkstiff(60, 59) =
        theta *
        (-(1. / V_arspl_sys_np) *
            pow(((dcb_c_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
                     (v_tissspl_ / V_arspl_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arspl_sys_np)) *
                        (dcb_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
                            (v_tissspl_ / V_arspl_sys_np) * dct_o2_dpp_o2(ppO2_arspl_sys_np)) -
                    dcb_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np)),
                -1.) *
            ((dcb_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
                 (v_tissspl_ / V_arspl_sys_np) * dct_o2_dpp_o2(ppO2_arspl_sys_np)) *
                    q_arspl_sys_in_np * dcb_c_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) -
                dcb_c_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) * q_arspl_sys_in_np *
                    dcb_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)));
    // w.r.t. ppCO2
    wkstiff(60, 60) =
        1. / ts_size +
        theta *
            (-(1. / V_arspl_sys_np) *
                    pow(((dcb_c_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
                             (v_tissspl_ / V_arspl_sys_np) *
                                 dct_c_o2_dpp_c_o2(ppCO2_arspl_sys_np)) *
                                (dcb_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
                                    (v_tissspl_ / V_arspl_sys_np) *
                                        dct_o2_dpp_o2(ppO2_arspl_sys_np)) -
                            dcb_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np)),
                        -2.) *
                    ((d2cb_c_o2_dpp_c_o22(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
                         (v_tissspl_ / V_arspl_sys_np) * d2ct_c_o2_dpp_c_o22(ppCO2_arspl_sys_np)) *
                            (dcb_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
                                (v_tissspl_ / V_arspl_sys_np) * dct_o2_dpp_o2(ppO2_arspl_sys_np)) +
                        (dcb_c_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
                            (v_tissspl_ / V_arspl_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arspl_sys_np)) *
                            d2cb_o2_dpp_o2dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) -
                        d2cb_o2_dpp_c_o22(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) *
                            dcb_c_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) -
                        dcb_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) *
                            d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np)) *
                    ((dcb_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
                         (v_tissspl_ / V_arspl_sys_np) * dct_o2_dpp_o2(ppO2_arspl_sys_np)) *
                            (q_arspl_sys_in_np * (cb_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) -
                                                     cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                                m_c_o2_arspl_) -
                        dcb_c_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) *
                            (q_arspl_sys_in_np * (cb_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) -
                                                     cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) +
                                m_o2_arspl_ * ct_o2(ppO2_arspl_sys_np) /
                                    (beta + ct_o2(ppO2_arspl_sys_np)))) +
                (1. / V_arspl_sys_np) *
                    pow(((dcb_c_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
                             (v_tissspl_ / V_arspl_sys_np) *
                                 dct_c_o2_dpp_c_o2(ppCO2_arspl_sys_np)) *
                                (dcb_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
                                    (v_tissspl_ / V_arspl_sys_np) *
                                        dct_o2_dpp_o2(ppO2_arspl_sys_np)) -
                            dcb_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np)),
                        -1.) *
                    (d2cb_o2_dpp_o2dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) *
                            (q_arspl_sys_in_np * (cb_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) -
                                                     cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                                m_c_o2_arspl_) +
                        (dcb_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
                            (v_tissspl_ / V_arspl_sys_np) * dct_o2_dpp_o2(ppO2_arspl_sys_np)) *
                            q_arspl_sys_in_np *
                            dcb_c_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) -
                        d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) *
                            (q_arspl_sys_in_np * (cb_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) -
                                                     cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) +
                                m_o2_arspl_ * ct_o2(ppO2_arspl_sys_np) /
                                    (beta + ct_o2(ppO2_arspl_sys_np))) -
                        dcb_c_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) * q_arspl_sys_in_np *
                            dcb_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np)));
    // w.r.t. ppO2
    wkstiff(60, 61) =
        theta *
        (-(1. / V_arspl_sys_np) *
                pow(((dcb_c_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
                         (v_tissspl_ / V_arspl_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arspl_sys_np)) *
                            (dcb_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
                                (v_tissspl_ / V_arspl_sys_np) * dct_o2_dpp_o2(ppO2_arspl_sys_np)) -
                        dcb_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) *
                            dcb_c_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np)),
                    -2.) *
                (d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) *
                        (dcb_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
                            (v_tissspl_ / V_arspl_sys_np) * dct_o2_dpp_o2(ppO2_arspl_sys_np)) +
                    (dcb_c_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
                        (v_tissspl_ / V_arspl_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arspl_sys_np)) *
                        (d2cb_o2_dpp_o22(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
                            (v_tissspl_ / V_arspl_sys_np) * d2ct_o2_dpp_o22(ppO2_arspl_sys_np)) -
                    d2cb_o2_dpp_o2dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) -
                    dcb_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) *
                        d2cb_c_o2_dpp_o22(ppCO2_arspl_sys_np, ppO2_arspl_sys_np)) *
                ((dcb_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
                     (v_tissspl_ / V_arspl_sys_np) * dct_o2_dpp_o2(ppO2_arspl_sys_np)) *
                        (q_arspl_sys_in_np * (cb_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) -
                                                 cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                            m_c_o2_arspl_) -
                    dcb_c_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) *
                        (q_arspl_sys_in_np * (cb_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) -
                                                 cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) +
                            m_o2_arspl_ * ct_o2(ppO2_arspl_sys_np) /
                                (beta + ct_o2(ppO2_arspl_sys_np)))) +
            (1. / V_arspl_sys_np) *
                pow(((dcb_c_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
                         (v_tissspl_ / V_arspl_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arspl_sys_np)) *
                            (dcb_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
                                (v_tissspl_ / V_arspl_sys_np) * dct_o2_dpp_o2(ppO2_arspl_sys_np)) -
                        dcb_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) *
                            dcb_c_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np)),
                    -1.) *
                ((d2cb_o2_dpp_o22(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
                     (v_tissspl_ / V_arspl_sys_np) * d2ct_o2_dpp_o22(ppO2_arspl_sys_np)) *
                        (q_arspl_sys_in_np * (cb_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) -
                                                 cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                            m_c_o2_arspl_) +
                    (dcb_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
                        (v_tissspl_ / V_arspl_sys_np) * dct_o2_dpp_o2(ppO2_arspl_sys_np)) *
                        q_arspl_sys_in_np * dcb_c_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) -
                    d2cb_c_o2_dpp_o22(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) *
                        (q_arspl_sys_in_np * (cb_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) -
                                                 cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) +
                            m_o2_arspl_ * ct_o2(ppO2_arspl_sys_np) /
                                (beta + ct_o2(ppO2_arspl_sys_np))) -
                    dcb_c_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) *
                        (q_arspl_sys_in_np * dcb_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
                            dct_o2_dpp_o2(ppO2_arspl_sys_np) * beta /
                                ((beta + ct_o2(ppO2_arspl_sys_np)) *
                                    (beta + ct_o2(ppO2_arspl_sys_np))))));

    //////// systemic splanchnic arteries O2
    // w.r.t. mech. pressure
    wkstiff(61, 6) =
        theta *
        (c_arspl_sys_ * (-1. / (V_arspl_sys_np * V_arspl_sys_np)) *
                pow(((dcb_c_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
                         (v_tissspl_ / V_arspl_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arspl_sys_np)) *
                            (dcb_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
                                (v_tissspl_ / V_arspl_sys_np) * dct_o2_dpp_o2(ppO2_arspl_sys_np)) -
                        dcb_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) *
                            dcb_c_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np)),
                    -1.) *
                ((dcb_c_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
                     (v_tissspl_ / V_arspl_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arspl_sys_np)) *
                        (q_arspl_sys_in_np * (cb_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) -
                                                 cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) +
                            m_o2_arspl_ * ct_o2(ppO2_arspl_sys_np) /
                                (beta + ct_o2(ppO2_arspl_sys_np))) -
                    dcb_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) *
                        (q_arspl_sys_in_np * (cb_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) -
                                                 cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                            m_c_o2_arspl_)) +
            (1. / V_arspl_sys_np) *
                (pow(((dcb_c_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
                          (v_tissspl_ / V_arspl_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arspl_sys_np)) *
                             (dcb_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
                                 (v_tissspl_ / V_arspl_sys_np) * dct_o2_dpp_o2(ppO2_arspl_sys_np)) -
                         dcb_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) *
                             dcb_c_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np)),
                    -1.)) *
                (c_arspl_sys_ * (-v_tissspl_ / (V_arspl_sys_np * V_arspl_sys_np)) *
                    dct_c_o2_dpp_c_o2(ppCO2_arspl_sys_np) *
                    (q_arspl_sys_in_np * (cb_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) -
                                             cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) +
                        m_o2_arspl_ * ct_o2(ppO2_arspl_sys_np) /
                            (beta + ct_o2(ppO2_arspl_sys_np))) *
                    (-pow(((dcb_c_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
                               (v_tissspl_ / V_arspl_sys_np) *
                                   dct_c_o2_dpp_c_o2(ppCO2_arspl_sys_np)) *
                                  (dcb_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
                                      (v_tissspl_ / V_arspl_sys_np) *
                                          dct_o2_dpp_o2(ppO2_arspl_sys_np)) -
                              dcb_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) *
                                  dcb_c_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np)),
                         -2.) *
                            (dcb_c_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
                                (v_tissspl_ / V_arspl_sys_np) *
                                    dct_c_o2_dpp_c_o2(ppCO2_arspl_sys_np)) *
                            c_arspl_sys_ * (-v_tissspl_ / (V_arspl_sys_np * V_arspl_sys_np)) *
                            dct_o2_dpp_o2(ppO2_arspl_sys_np) +
                        c_arspl_sys_ * (-v_tissspl_ / (V_arspl_sys_np * V_arspl_sys_np)) *
                            dct_c_o2_dpp_c_o2(ppCO2_arspl_sys_np) *
                            (dcb_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
                                (v_tissspl_ / V_arspl_sys_np) * dct_o2_dpp_o2(ppO2_arspl_sys_np))) *
                    ((dcb_c_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
                         (v_tissspl_ / V_arspl_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arspl_sys_np)) *
                            (q_arspl_sys_in_np * (cb_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) -
                                                     cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) +
                                m_o2_arspl_ * ct_o2(ppO2_arspl_sys_np) /
                                    (beta + ct_o2(ppO2_arspl_sys_np))) -
                        dcb_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) *
                            (q_arspl_sys_in_np * (cb_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) -
                                                     cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                                m_c_o2_arspl_))));
    // w.r.t. upstream flux
    wkstiff(61, 39) =
        theta *
        ((1. / V_arspl_sys_np) *
            pow(((dcb_c_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
                     (v_tissspl_ / V_arspl_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arspl_sys_np)) *
                        (dcb_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
                            (v_tissspl_ / V_arspl_sys_np) * dct_o2_dpp_o2(ppO2_arspl_sys_np)) -
                    dcb_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np)),
                -1.) *
            (dcb_c_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) * 1.0 *
                    (cb_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) -
                        cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                dcb_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) * 1.0 *
                    (cb_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) -
                        cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np))));
    // w.r.t. upstream ppCO2
    wkstiff(61, 58) =
        theta *
        (-(1. / V_arspl_sys_np) *
            pow(((dcb_c_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
                     (v_tissspl_ / V_arspl_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arspl_sys_np)) *
                        (dcb_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
                            (v_tissspl_ / V_arspl_sys_np) * dct_o2_dpp_o2(ppO2_arspl_sys_np)) -
                    dcb_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np)),
                -1.) *
            ((dcb_c_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
                 (v_tissspl_ / V_arspl_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arspl_sys_np)) *
                    q_arspl_sys_in_np * dcb_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) -
                dcb_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) * q_arspl_sys_in_np *
                    dcb_c_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)));
    // w.r.t. upstream ppO2
    wkstiff(61, 59) =
        theta *
        (-(1. / V_arspl_sys_np) *
            pow(((dcb_c_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
                     (v_tissspl_ / V_arspl_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arspl_sys_np)) *
                        (dcb_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
                            (v_tissspl_ / V_arspl_sys_np) * dct_o2_dpp_o2(ppO2_arspl_sys_np)) -
                    dcb_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np)),
                -1.) *
            ((dcb_c_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
                 (v_tissspl_ / V_arspl_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arspl_sys_np)) *
                    q_arspl_sys_in_np * dcb_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) -
                dcb_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) * q_arspl_sys_in_np *
                    dcb_c_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)));
    // w.r.t. ppCO2
    wkstiff(61, 60) =
        theta *
        (-(1. / V_arspl_sys_np) *
                pow(((dcb_c_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
                         (v_tissspl_ / V_arspl_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arspl_sys_np)) *
                            (dcb_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
                                (v_tissspl_ / V_arspl_sys_np) * dct_o2_dpp_o2(ppO2_arspl_sys_np)) -
                        dcb_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) *
                            dcb_c_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np)),
                    -2.) *
                ((d2cb_c_o2_dpp_c_o22(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
                     (v_tissspl_ / V_arspl_sys_np) * d2ct_c_o2_dpp_c_o22(ppCO2_arspl_sys_np)) *
                        (dcb_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
                            (v_tissspl_ / V_arspl_sys_np) * dct_o2_dpp_o2(ppO2_arspl_sys_np)) +
                    (dcb_c_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
                        (v_tissspl_ / V_arspl_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arspl_sys_np)) *
                        d2cb_o2_dpp_o2dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) -
                    d2cb_o2_dpp_c_o22(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) -
                    dcb_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) *
                        d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np)) *
                ((dcb_c_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
                     (v_tissspl_ / V_arspl_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arspl_sys_np)) *
                        (q_arspl_sys_in_np * (cb_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) -
                                                 cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) +
                            m_o2_arspl_ * ct_o2(ppO2_arspl_sys_np) /
                                (beta + ct_o2(ppO2_arspl_sys_np))) -
                    dcb_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) *
                        (q_arspl_sys_in_np * (cb_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) -
                                                 cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                            m_c_o2_arspl_)) +
            (1. / V_arspl_sys_np) *
                pow(((dcb_c_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
                         (v_tissspl_ / V_arspl_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arspl_sys_np)) *
                            (dcb_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
                                (v_tissspl_ / V_arspl_sys_np) * dct_o2_dpp_o2(ppO2_arspl_sys_np)) -
                        dcb_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) *
                            dcb_c_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np)),
                    -1.) *
                ((d2cb_c_o2_dpp_c_o22(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
                     (v_tissspl_ / V_arspl_sys_np) * d2ct_c_o2_dpp_c_o22(ppCO2_arspl_sys_np)) *
                        (q_arspl_sys_in_np * (cb_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) -
                                                 cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) +
                            m_o2_arspl_ * ct_o2(ppO2_arspl_sys_np) /
                                (beta + ct_o2(ppO2_arspl_sys_np))) +
                    (dcb_c_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
                        (v_tissspl_ / V_arspl_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arspl_sys_np)) *
                        q_arspl_sys_in_np * dcb_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) -
                    d2cb_o2_dpp_c_o22(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) *
                        (q_arspl_sys_in_np * (cb_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) -
                                                 cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                            m_c_o2_arspl_) -
                    dcb_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) * q_arspl_sys_in_np *
                        dcb_c_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np)));
    // w.r.t. ppO2
    wkstiff(61, 61) =
        1. / ts_size +
        theta *
            (-(1. / V_arspl_sys_np) *
                    pow(((dcb_c_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
                             (v_tissspl_ / V_arspl_sys_np) *
                                 dct_c_o2_dpp_c_o2(ppCO2_arspl_sys_np)) *
                                (dcb_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
                                    (v_tissspl_ / V_arspl_sys_np) *
                                        dct_o2_dpp_o2(ppO2_arspl_sys_np)) -
                            dcb_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np)),
                        -2.) *
                    (d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) *
                            (dcb_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
                                (v_tissspl_ / V_arspl_sys_np) * dct_o2_dpp_o2(ppO2_arspl_sys_np)) +
                        (dcb_c_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
                            (v_tissspl_ / V_arspl_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arspl_sys_np)) *
                            (d2cb_o2_dpp_o22(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
                                (v_tissspl_ / V_arspl_sys_np) *
                                    d2ct_o2_dpp_o22(ppO2_arspl_sys_np)) -
                        d2cb_o2_dpp_o2dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) *
                            dcb_c_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) -
                        dcb_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) *
                            d2cb_c_o2_dpp_o22(ppCO2_arspl_sys_np, ppO2_arspl_sys_np)) *
                    ((dcb_c_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
                         (v_tissspl_ / V_arspl_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arspl_sys_np)) *
                            (q_arspl_sys_in_np * (cb_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) -
                                                     cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) +
                                m_o2_arspl_ * ct_o2(ppO2_arspl_sys_np) /
                                    (beta + ct_o2(ppO2_arspl_sys_np))) -
                        dcb_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) *
                            (q_arspl_sys_in_np * (cb_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) -
                                                     cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                                m_c_o2_arspl_)) +
                (1. / V_arspl_sys_np) *
                    pow(((dcb_c_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
                             (v_tissspl_ / V_arspl_sys_np) *
                                 dct_c_o2_dpp_c_o2(ppCO2_arspl_sys_np)) *
                                (dcb_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
                                    (v_tissspl_ / V_arspl_sys_np) *
                                        dct_o2_dpp_o2(ppO2_arspl_sys_np)) -
                            dcb_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np)),
                        -1.) *
                    (d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) *
                            (q_arspl_sys_in_np * (cb_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) -
                                                     cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) +
                                m_o2_arspl_ * ct_o2(ppO2_arspl_sys_np) /
                                    (beta + ct_o2(ppO2_arspl_sys_np))) +
                        (dcb_c_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
                            (v_tissspl_ / V_arspl_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arspl_sys_np)) *
                            (q_arspl_sys_in_np *
                                    dcb_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) +
                                dct_o2_dpp_o2(ppO2_arspl_sys_np) * beta /
                                    ((beta + ct_o2(ppO2_arspl_sys_np)) *
                                        (beta + ct_o2(ppO2_arspl_sys_np)))) -
                        d2cb_o2_dpp_o2dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) *
                            (q_arspl_sys_in_np * (cb_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) -
                                                     cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                                m_c_o2_arspl_) -
                        dcb_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) * q_arspl_sys_in_np *
                            dcb_c_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np)));


    //////// systemic extra-splanchnic arteries CO2
    // w.r.t. mech. pressure
    wkstiff(62, 6) =
        theta *
        (c_arespl_sys_ * (-1. / (V_arespl_sys_np * V_arespl_sys_np)) *
                pow(((dcb_c_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
                         (v_tissespl_ / V_arespl_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arespl_sys_np)) *
                            (dcb_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
                                (v_tissespl_ / V_arespl_sys_np) *
                                    dct_o2_dpp_o2(ppO2_arespl_sys_np)) -
                        dcb_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) *
                            dcb_c_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np)),
                    -1.) *
                ((dcb_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
                     (v_tissespl_ / V_arespl_sys_np) * dct_o2_dpp_o2(ppO2_arespl_sys_np)) *
                        (q_arespl_sys_in_np * (cb_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) -
                                                  cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                            m_c_o2_arespl_) -
                    dcb_c_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) *
                        (q_arespl_sys_in_np * (cb_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) -
                                                  cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) +
                            m_o2_arespl_ * ct_o2(ppO2_arespl_sys_np) /
                                (beta + ct_o2(ppO2_arespl_sys_np)))) +
            (1. / V_arespl_sys_np) *
                (pow(
                    ((dcb_c_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
                         (v_tissespl_ / V_arespl_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arespl_sys_np)) *
                            (dcb_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
                                (v_tissespl_ / V_arespl_sys_np) *
                                    dct_o2_dpp_o2(ppO2_arespl_sys_np)) -
                        dcb_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) *
                            dcb_c_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np)),
                    -1.)) *
                (c_arespl_sys_ * (-v_tissespl_ / (V_arespl_sys_np * V_arespl_sys_np)) *
                    dct_o2_dpp_o2(ppO2_arespl_sys_np) *
                    (q_arespl_sys_in_np * (cb_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) -
                                              cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                        m_c_o2_arespl_) *
                    (-pow(((dcb_c_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
                               (v_tissespl_ / V_arespl_sys_np) *
                                   dct_c_o2_dpp_c_o2(ppCO2_arespl_sys_np)) *
                                  (dcb_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
                                      (v_tissespl_ / V_arespl_sys_np) *
                                          dct_o2_dpp_o2(ppO2_arespl_sys_np)) -
                              dcb_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) *
                                  dcb_c_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np)),
                         -2.) *
                            (dcb_c_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
                                (v_tissespl_ / V_arespl_sys_np) *
                                    dct_c_o2_dpp_c_o2(ppCO2_arespl_sys_np)) *
                            c_arespl_sys_ * (-v_tissespl_ / (V_arespl_sys_np * V_arespl_sys_np)) *
                            dct_o2_dpp_o2(ppO2_arespl_sys_np) +
                        c_arespl_sys_ * (-v_tissespl_ / (V_arespl_sys_np * V_arespl_sys_np)) *
                            dct_c_o2_dpp_c_o2(ppCO2_arespl_sys_np) *
                            (dcb_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
                                (v_tissespl_ / V_arespl_sys_np) *
                                    dct_o2_dpp_o2(ppO2_arespl_sys_np))) *
                    ((dcb_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
                         (v_tissespl_ / V_arespl_sys_np) * dct_o2_dpp_o2(ppO2_arespl_sys_np)) *
                            (q_arespl_sys_in_np *
                                    (cb_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) -
                                        cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                                m_c_o2_arespl_) -
                        dcb_c_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) *
                            (q_arespl_sys_in_np * (cb_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) -
                                                      cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) +
                                m_o2_arespl_ * ct_o2(ppO2_arespl_sys_np) /
                                    (beta + ct_o2(ppO2_arespl_sys_np))))));
    // w.r.t. upstream flux
    wkstiff(62, 40) =
        theta *
        ((1. / V_arespl_sys_np) *
            pow(((dcb_c_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
                     (v_tissespl_ / V_arespl_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arespl_sys_np)) *
                        (dcb_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
                            (v_tissespl_ / V_arespl_sys_np) * dct_o2_dpp_o2(ppO2_arespl_sys_np)) -
                    dcb_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np)),
                -1.) *
            (dcb_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) * 1.0 *
                    (cb_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) -
                        cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                dcb_c_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) * 1.0 *
                    (cb_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) -
                        cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np))));
    // w.r.t. upstream ppCO2
    wkstiff(62, 58) =
        theta *
        (-(1. / V_arespl_sys_np) *
            pow(((dcb_c_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
                     (v_tissespl_ / V_arespl_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arespl_sys_np)) *
                        (dcb_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
                            (v_tissespl_ / V_arespl_sys_np) * dct_o2_dpp_o2(ppO2_arespl_sys_np)) -
                    dcb_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np)),
                -1.) *
            ((dcb_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
                 (v_tissespl_ / V_arespl_sys_np) * dct_o2_dpp_o2(ppO2_arespl_sys_np)) *
                    q_arespl_sys_in_np * dcb_c_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) -
                dcb_c_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) * q_arespl_sys_in_np *
                    dcb_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)));
    // w.r.t. upstream ppO2
    wkstiff(62, 59) =
        theta *
        (-(1. / V_arespl_sys_np) *
            pow(((dcb_c_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
                     (v_tissespl_ / V_arespl_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arespl_sys_np)) *
                        (dcb_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
                            (v_tissespl_ / V_arespl_sys_np) * dct_o2_dpp_o2(ppO2_arespl_sys_np)) -
                    dcb_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np)),
                -1.) *
            ((dcb_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
                 (v_tissespl_ / V_arespl_sys_np) * dct_o2_dpp_o2(ppO2_arespl_sys_np)) *
                    q_arespl_sys_in_np * dcb_c_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) -
                dcb_c_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) * q_arespl_sys_in_np *
                    dcb_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)));
    // w.r.t. ppCO2
    wkstiff(62, 62) =
        1. / ts_size +
        theta *
            (-(1. / V_arespl_sys_np) *
                    pow(((dcb_c_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
                             (v_tissespl_ / V_arespl_sys_np) *
                                 dct_c_o2_dpp_c_o2(ppCO2_arespl_sys_np)) *
                                (dcb_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
                                    (v_tissespl_ / V_arespl_sys_np) *
                                        dct_o2_dpp_o2(ppO2_arespl_sys_np)) -
                            dcb_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np)),
                        -2.) *
                    ((d2cb_c_o2_dpp_c_o22(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
                         (v_tissespl_ / V_arespl_sys_np) *
                             d2ct_c_o2_dpp_c_o22(ppCO2_arespl_sys_np)) *
                            (dcb_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
                                (v_tissespl_ / V_arespl_sys_np) *
                                    dct_o2_dpp_o2(ppO2_arespl_sys_np)) +
                        (dcb_c_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
                            (v_tissespl_ / V_arespl_sys_np) *
                                dct_c_o2_dpp_c_o2(ppCO2_arespl_sys_np)) *
                            d2cb_o2_dpp_o2dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) -
                        d2cb_o2_dpp_c_o22(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) *
                            dcb_c_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) -
                        dcb_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) *
                            d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np)) *
                    ((dcb_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
                         (v_tissespl_ / V_arespl_sys_np) * dct_o2_dpp_o2(ppO2_arespl_sys_np)) *
                            (q_arespl_sys_in_np *
                                    (cb_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) -
                                        cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                                m_c_o2_arespl_) -
                        dcb_c_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) *
                            (q_arespl_sys_in_np * (cb_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) -
                                                      cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) +
                                m_o2_arespl_ * ct_o2(ppO2_arespl_sys_np) /
                                    (beta + ct_o2(ppO2_arespl_sys_np)))) +
                (1. / V_arespl_sys_np) *
                    pow(((dcb_c_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
                             (v_tissespl_ / V_arespl_sys_np) *
                                 dct_c_o2_dpp_c_o2(ppCO2_arespl_sys_np)) *
                                (dcb_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
                                    (v_tissespl_ / V_arespl_sys_np) *
                                        dct_o2_dpp_o2(ppO2_arespl_sys_np)) -
                            dcb_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np)),
                        -1.) *
                    (d2cb_o2_dpp_o2dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) *
                            (q_arespl_sys_in_np *
                                    (cb_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) -
                                        cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                                m_c_o2_arespl_) +
                        (dcb_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
                            (v_tissespl_ / V_arespl_sys_np) * dct_o2_dpp_o2(ppO2_arespl_sys_np)) *
                            q_arespl_sys_in_np *
                            dcb_c_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) -
                        d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) *
                            (q_arespl_sys_in_np * (cb_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) -
                                                      cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) +
                                m_o2_arespl_ * ct_o2(ppO2_arespl_sys_np) /
                                    (beta + ct_o2(ppO2_arespl_sys_np))) -
                        dcb_c_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) *
                            q_arespl_sys_in_np *
                            dcb_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np)));
    // w.r.t. ppO2
    wkstiff(62, 63) =
        theta *
        (-(1. / V_arespl_sys_np) *
                pow(((dcb_c_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
                         (v_tissespl_ / V_arespl_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arespl_sys_np)) *
                            (dcb_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
                                (v_tissespl_ / V_arespl_sys_np) *
                                    dct_o2_dpp_o2(ppO2_arespl_sys_np)) -
                        dcb_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) *
                            dcb_c_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np)),
                    -2.) *
                (d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) *
                        (dcb_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
                            (v_tissespl_ / V_arespl_sys_np) * dct_o2_dpp_o2(ppO2_arespl_sys_np)) +
                    (dcb_c_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
                        (v_tissespl_ / V_arespl_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arespl_sys_np)) *
                        (d2cb_o2_dpp_o22(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
                            (v_tissespl_ / V_arespl_sys_np) * d2ct_o2_dpp_o22(ppO2_arespl_sys_np)) -
                    d2cb_o2_dpp_o2dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) -
                    dcb_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) *
                        d2cb_c_o2_dpp_o22(ppCO2_arespl_sys_np, ppO2_arespl_sys_np)) *
                ((dcb_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
                     (v_tissespl_ / V_arespl_sys_np) * dct_o2_dpp_o2(ppO2_arespl_sys_np)) *
                        (q_arespl_sys_in_np * (cb_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) -
                                                  cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                            m_c_o2_arespl_) -
                    dcb_c_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) *
                        (q_arespl_sys_in_np * (cb_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) -
                                                  cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) +
                            m_o2_arespl_ * ct_o2(ppO2_arespl_sys_np) /
                                (beta + ct_o2(ppO2_arespl_sys_np)))) +
            (1. / V_arespl_sys_np) *
                pow(((dcb_c_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
                         (v_tissespl_ / V_arespl_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arespl_sys_np)) *
                            (dcb_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
                                (v_tissespl_ / V_arespl_sys_np) *
                                    dct_o2_dpp_o2(ppO2_arespl_sys_np)) -
                        dcb_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) *
                            dcb_c_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np)),
                    -1.) *
                ((d2cb_o2_dpp_o22(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
                     (v_tissespl_ / V_arespl_sys_np) * d2ct_o2_dpp_o22(ppO2_arespl_sys_np)) *
                        (q_arespl_sys_in_np * (cb_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) -
                                                  cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                            m_c_o2_arespl_) +
                    (dcb_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
                        (v_tissespl_ / V_arespl_sys_np) * dct_o2_dpp_o2(ppO2_arespl_sys_np)) *
                        q_arespl_sys_in_np *
                        dcb_c_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) -
                    d2cb_c_o2_dpp_o22(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) *
                        (q_arespl_sys_in_np * (cb_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) -
                                                  cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) +
                            m_o2_arespl_ * ct_o2(ppO2_arespl_sys_np) /
                                (beta + ct_o2(ppO2_arespl_sys_np))) -
                    dcb_c_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) *
                        (q_arespl_sys_in_np *
                                dcb_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
                            dct_o2_dpp_o2(ppO2_arespl_sys_np) * beta /
                                ((beta + ct_o2(ppO2_arespl_sys_np)) *
                                    (beta + ct_o2(ppO2_arespl_sys_np))))));

    //////// systemic extra-splanchnic arteries O2
    // w.r.t. mech. pressure
    wkstiff(63, 6) =
        theta *
        (c_arespl_sys_ * (-1. / (V_arespl_sys_np * V_arespl_sys_np)) *
                pow(((dcb_c_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
                         (v_tissespl_ / V_arespl_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arespl_sys_np)) *
                            (dcb_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
                                (v_tissespl_ / V_arespl_sys_np) *
                                    dct_o2_dpp_o2(ppO2_arespl_sys_np)) -
                        dcb_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) *
                            dcb_c_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np)),
                    -1.) *
                ((dcb_c_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
                     (v_tissespl_ / V_arespl_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arespl_sys_np)) *
                        (q_arespl_sys_in_np * (cb_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) -
                                                  cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) +
                            m_o2_arespl_ * ct_o2(ppO2_arespl_sys_np) /
                                (beta + ct_o2(ppO2_arespl_sys_np))) -
                    dcb_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) *
                        (q_arespl_sys_in_np * (cb_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) -
                                                  cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                            m_c_o2_arespl_)) +
            (1. / V_arespl_sys_np) *
                (pow(
                    ((dcb_c_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
                         (v_tissespl_ / V_arespl_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arespl_sys_np)) *
                            (dcb_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
                                (v_tissespl_ / V_arespl_sys_np) *
                                    dct_o2_dpp_o2(ppO2_arespl_sys_np)) -
                        dcb_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) *
                            dcb_c_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np)),
                    -1.)) *
                (c_arespl_sys_ * (-v_tissespl_ / (V_arespl_sys_np * V_arespl_sys_np)) *
                    dct_c_o2_dpp_c_o2(ppCO2_arespl_sys_np) *
                    (q_arespl_sys_in_np * (cb_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) -
                                              cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) +
                        m_o2_arespl_ * ct_o2(ppO2_arespl_sys_np) /
                            (beta + ct_o2(ppO2_arespl_sys_np))) *
                    (-pow(((dcb_c_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
                               (v_tissespl_ / V_arespl_sys_np) *
                                   dct_c_o2_dpp_c_o2(ppCO2_arespl_sys_np)) *
                                  (dcb_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
                                      (v_tissespl_ / V_arespl_sys_np) *
                                          dct_o2_dpp_o2(ppO2_arespl_sys_np)) -
                              dcb_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) *
                                  dcb_c_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np)),
                         -2.) *
                            (dcb_c_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
                                (v_tissespl_ / V_arespl_sys_np) *
                                    dct_c_o2_dpp_c_o2(ppCO2_arespl_sys_np)) *
                            c_arespl_sys_ * (-v_tissespl_ / (V_arespl_sys_np * V_arespl_sys_np)) *
                            dct_o2_dpp_o2(ppO2_arespl_sys_np) +
                        c_arespl_sys_ * (-v_tissespl_ / (V_arespl_sys_np * V_arespl_sys_np)) *
                            dct_c_o2_dpp_c_o2(ppCO2_arespl_sys_np) *
                            (dcb_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
                                (v_tissespl_ / V_arespl_sys_np) *
                                    dct_o2_dpp_o2(ppO2_arespl_sys_np))) *
                    ((dcb_c_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
                         (v_tissespl_ / V_arespl_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arespl_sys_np)) *
                            (q_arespl_sys_in_np * (cb_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) -
                                                      cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) +
                                m_o2_arespl_ * ct_o2(ppO2_arespl_sys_np) /
                                    (beta + ct_o2(ppO2_arespl_sys_np))) -
                        dcb_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) *
                            (q_arespl_sys_in_np *
                                    (cb_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) -
                                        cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                                m_c_o2_arespl_))));
    // w.r.t. upstream flux
    wkstiff(63, 40) =
        theta *
        ((1. / V_arespl_sys_np) *
            pow(((dcb_c_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
                     (v_tissespl_ / V_arespl_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arespl_sys_np)) *
                        (dcb_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
                            (v_tissespl_ / V_arespl_sys_np) * dct_o2_dpp_o2(ppO2_arespl_sys_np)) -
                    dcb_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np)),
                -1.) *
            (dcb_c_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) * 1.0 *
                    (cb_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) -
                        cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                dcb_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) * 1.0 *
                    (cb_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) -
                        cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np))));
    // w.r.t. upstream ppCO2
    wkstiff(63, 58) =
        theta *
        (-(1. / V_arespl_sys_np) *
            pow(((dcb_c_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
                     (v_tissespl_ / V_arespl_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arespl_sys_np)) *
                        (dcb_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
                            (v_tissespl_ / V_arespl_sys_np) * dct_o2_dpp_o2(ppO2_arespl_sys_np)) -
                    dcb_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np)),
                -1.) *
            ((dcb_c_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
                 (v_tissespl_ / V_arespl_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arespl_sys_np)) *
                    q_arespl_sys_in_np * dcb_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) -
                dcb_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) * q_arespl_sys_in_np *
                    dcb_c_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)));
    // w.r.t. upstream ppO2
    wkstiff(63, 59) =
        theta *
        (-(1. / V_arespl_sys_np) *
            pow(((dcb_c_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
                     (v_tissespl_ / V_arespl_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arespl_sys_np)) *
                        (dcb_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
                            (v_tissespl_ / V_arespl_sys_np) * dct_o2_dpp_o2(ppO2_arespl_sys_np)) -
                    dcb_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np)),
                -1.) *
            ((dcb_c_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
                 (v_tissespl_ / V_arespl_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arespl_sys_np)) *
                    q_arespl_sys_in_np * dcb_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) -
                dcb_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) * q_arespl_sys_in_np *
                    dcb_c_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)));
    // w.r.t. ppCO2
    wkstiff(63, 62) =
        theta *
        (-(1. / V_arespl_sys_np) *
                pow(((dcb_c_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
                         (v_tissespl_ / V_arespl_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arespl_sys_np)) *
                            (dcb_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
                                (v_tissespl_ / V_arespl_sys_np) *
                                    dct_o2_dpp_o2(ppO2_arespl_sys_np)) -
                        dcb_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) *
                            dcb_c_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np)),
                    -2.) *
                ((d2cb_c_o2_dpp_c_o22(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
                     (v_tissespl_ / V_arespl_sys_np) * d2ct_c_o2_dpp_c_o22(ppCO2_arespl_sys_np)) *
                        (dcb_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
                            (v_tissespl_ / V_arespl_sys_np) * dct_o2_dpp_o2(ppO2_arespl_sys_np)) +
                    (dcb_c_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
                        (v_tissespl_ / V_arespl_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arespl_sys_np)) *
                        d2cb_o2_dpp_o2dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) -
                    d2cb_o2_dpp_c_o22(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) -
                    dcb_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) *
                        d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np)) *
                ((dcb_c_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
                     (v_tissespl_ / V_arespl_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arespl_sys_np)) *
                        (q_arespl_sys_in_np * (cb_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) -
                                                  cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) +
                            m_o2_arespl_ * ct_o2(ppO2_arespl_sys_np) /
                                (beta + ct_o2(ppO2_arespl_sys_np))) -
                    dcb_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) *
                        (q_arespl_sys_in_np * (cb_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) -
                                                  cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                            m_c_o2_arespl_)) +
            (1. / V_arespl_sys_np) *
                pow(((dcb_c_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
                         (v_tissespl_ / V_arespl_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arespl_sys_np)) *
                            (dcb_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
                                (v_tissespl_ / V_arespl_sys_np) *
                                    dct_o2_dpp_o2(ppO2_arespl_sys_np)) -
                        dcb_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) *
                            dcb_c_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np)),
                    -1.) *
                ((d2cb_c_o2_dpp_c_o22(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
                     (v_tissespl_ / V_arespl_sys_np) * d2ct_c_o2_dpp_c_o22(ppCO2_arespl_sys_np)) *
                        (q_arespl_sys_in_np * (cb_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) -
                                                  cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) +
                            m_o2_arespl_ * ct_o2(ppO2_arespl_sys_np) /
                                (beta + ct_o2(ppO2_arespl_sys_np))) +
                    (dcb_c_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
                        (v_tissespl_ / V_arespl_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arespl_sys_np)) *
                        q_arespl_sys_in_np *
                        dcb_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) -
                    d2cb_o2_dpp_c_o22(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) *
                        (q_arespl_sys_in_np * (cb_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) -
                                                  cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                            m_c_o2_arespl_) -
                    dcb_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) * q_arespl_sys_in_np *
                        dcb_c_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np)));
    // w.r.t. ppO2
    wkstiff(63, 63) =
        1. / ts_size +
        theta *
            (-(1. / V_arespl_sys_np) *
                    pow(((dcb_c_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
                             (v_tissespl_ / V_arespl_sys_np) *
                                 dct_c_o2_dpp_c_o2(ppCO2_arespl_sys_np)) *
                                (dcb_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
                                    (v_tissespl_ / V_arespl_sys_np) *
                                        dct_o2_dpp_o2(ppO2_arespl_sys_np)) -
                            dcb_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np)),
                        -2.) *
                    (d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) *
                            (dcb_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
                                (v_tissespl_ / V_arespl_sys_np) *
                                    dct_o2_dpp_o2(ppO2_arespl_sys_np)) +
                        (dcb_c_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
                            (v_tissespl_ / V_arespl_sys_np) *
                                dct_c_o2_dpp_c_o2(ppCO2_arespl_sys_np)) *
                            (d2cb_o2_dpp_o22(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
                                (v_tissespl_ / V_arespl_sys_np) *
                                    d2ct_o2_dpp_o22(ppO2_arespl_sys_np)) -
                        d2cb_o2_dpp_o2dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) *
                            dcb_c_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) -
                        dcb_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) *
                            d2cb_c_o2_dpp_o22(ppCO2_arespl_sys_np, ppO2_arespl_sys_np)) *
                    ((dcb_c_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
                         (v_tissespl_ / V_arespl_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arespl_sys_np)) *
                            (q_arespl_sys_in_np * (cb_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) -
                                                      cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) +
                                m_o2_arespl_ * ct_o2(ppO2_arespl_sys_np) /
                                    (beta + ct_o2(ppO2_arespl_sys_np))) -
                        dcb_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) *
                            (q_arespl_sys_in_np *
                                    (cb_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) -
                                        cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                                m_c_o2_arespl_)) +
                (1. / V_arespl_sys_np) *
                    pow(((dcb_c_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
                             (v_tissespl_ / V_arespl_sys_np) *
                                 dct_c_o2_dpp_c_o2(ppCO2_arespl_sys_np)) *
                                (dcb_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
                                    (v_tissespl_ / V_arespl_sys_np) *
                                        dct_o2_dpp_o2(ppO2_arespl_sys_np)) -
                            dcb_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np)),
                        -1.) *
                    (d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) *
                            (q_arespl_sys_in_np * (cb_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) -
                                                      cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) +
                                m_o2_arespl_ * ct_o2(ppO2_arespl_sys_np) /
                                    (beta + ct_o2(ppO2_arespl_sys_np))) +
                        (dcb_c_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
                            (v_tissespl_ / V_arespl_sys_np) *
                                dct_c_o2_dpp_c_o2(ppCO2_arespl_sys_np)) *
                            (q_arespl_sys_in_np *
                                    dcb_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) +
                                dct_o2_dpp_o2(ppO2_arespl_sys_np) * beta /
                                    ((beta + ct_o2(ppO2_arespl_sys_np)) *
                                        (beta + ct_o2(ppO2_arespl_sys_np)))) -
                        d2cb_o2_dpp_o2dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) *
                            (q_arespl_sys_in_np *
                                    (cb_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) -
                                        cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                                m_c_o2_arespl_) -
                        dcb_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) *
                            q_arespl_sys_in_np *
                            dcb_c_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np)));


    //////// systemic muscular arteries CO2
    // w.r.t. mech. pressure
    wkstiff(64, 6) =
        theta *
        (c_armsc_sys_ * (-1. / (V_armsc_sys_np * V_armsc_sys_np)) *
                pow(((dcb_c_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
                         (v_tissmsc_ / V_armsc_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_armsc_sys_np)) *
                            (dcb_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
                                (v_tissmsc_ / V_armsc_sys_np) * dct_o2_dpp_o2(ppO2_armsc_sys_np)) -
                        dcb_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) *
                            dcb_c_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np)),
                    -1.) *
                ((dcb_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
                     (v_tissmsc_ / V_armsc_sys_np) * dct_o2_dpp_o2(ppO2_armsc_sys_np)) *
                        (q_armsc_sys_in_np * (cb_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) -
                                                 cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                            m_c_o2_armsc_) -
                    dcb_c_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) *
                        (q_armsc_sys_in_np * (cb_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) -
                                                 cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) +
                            m_o2_armsc_ * ct_o2(ppO2_armsc_sys_np) /
                                (beta + ct_o2(ppO2_armsc_sys_np)))) +
            (1. / V_armsc_sys_np) *
                (pow(((dcb_c_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
                          (v_tissmsc_ / V_armsc_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_armsc_sys_np)) *
                             (dcb_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
                                 (v_tissmsc_ / V_armsc_sys_np) * dct_o2_dpp_o2(ppO2_armsc_sys_np)) -
                         dcb_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) *
                             dcb_c_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np)),
                    -1.)) *
                (c_armsc_sys_ * (-v_tissmsc_ / (V_armsc_sys_np * V_armsc_sys_np)) *
                    dct_o2_dpp_o2(ppO2_armsc_sys_np) *
                    (q_armsc_sys_in_np * (cb_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) -
                                             cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                        m_c_o2_armsc_) *
                    (-pow(((dcb_c_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
                               (v_tissmsc_ / V_armsc_sys_np) *
                                   dct_c_o2_dpp_c_o2(ppCO2_armsc_sys_np)) *
                                  (dcb_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
                                      (v_tissmsc_ / V_armsc_sys_np) *
                                          dct_o2_dpp_o2(ppO2_armsc_sys_np)) -
                              dcb_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) *
                                  dcb_c_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np)),
                         -2.) *
                            (dcb_c_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
                                (v_tissmsc_ / V_armsc_sys_np) *
                                    dct_c_o2_dpp_c_o2(ppCO2_armsc_sys_np)) *
                            c_armsc_sys_ * (-v_tissmsc_ / (V_armsc_sys_np * V_armsc_sys_np)) *
                            dct_o2_dpp_o2(ppO2_armsc_sys_np) +
                        c_armsc_sys_ * (-v_tissmsc_ / (V_armsc_sys_np * V_armsc_sys_np)) *
                            dct_c_o2_dpp_c_o2(ppCO2_armsc_sys_np) *
                            (dcb_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
                                (v_tissmsc_ / V_armsc_sys_np) * dct_o2_dpp_o2(ppO2_armsc_sys_np))) *
                    ((dcb_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
                         (v_tissmsc_ / V_armsc_sys_np) * dct_o2_dpp_o2(ppO2_armsc_sys_np)) *
                            (q_armsc_sys_in_np * (cb_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) -
                                                     cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                                m_c_o2_armsc_) -
                        dcb_c_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) *
                            (q_armsc_sys_in_np * (cb_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) -
                                                     cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) +
                                m_o2_armsc_ * ct_o2(ppO2_armsc_sys_np) /
                                    (beta + ct_o2(ppO2_armsc_sys_np))))));
    // w.r.t. upstream flux
    wkstiff(64, 41) =
        theta *
        ((1. / V_armsc_sys_np) *
            pow(((dcb_c_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
                     (v_tissmsc_ / V_armsc_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_armsc_sys_np)) *
                        (dcb_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
                            (v_tissmsc_ / V_armsc_sys_np) * dct_o2_dpp_o2(ppO2_armsc_sys_np)) -
                    dcb_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np)),
                -1.) *
            (dcb_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) * 1.0 *
                    (cb_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) -
                        cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                dcb_c_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) * 1.0 *
                    (cb_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) -
                        cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np))));
    // w.r.t. upstream ppCO2
    wkstiff(64, 58) =
        theta *
        (-(1. / V_armsc_sys_np) *
            pow(((dcb_c_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
                     (v_tissmsc_ / V_armsc_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_armsc_sys_np)) *
                        (dcb_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
                            (v_tissmsc_ / V_armsc_sys_np) * dct_o2_dpp_o2(ppO2_armsc_sys_np)) -
                    dcb_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np)),
                -1.) *
            ((dcb_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
                 (v_tissmsc_ / V_armsc_sys_np) * dct_o2_dpp_o2(ppO2_armsc_sys_np)) *
                    q_armsc_sys_in_np * dcb_c_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) -
                dcb_c_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) * q_armsc_sys_in_np *
                    dcb_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)));
    // w.r.t. upstream ppO2
    wkstiff(64, 59) =
        theta *
        (-(1. / V_armsc_sys_np) *
            pow(((dcb_c_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
                     (v_tissmsc_ / V_armsc_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_armsc_sys_np)) *
                        (dcb_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
                            (v_tissmsc_ / V_armsc_sys_np) * dct_o2_dpp_o2(ppO2_armsc_sys_np)) -
                    dcb_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np)),
                -1.) *
            ((dcb_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
                 (v_tissmsc_ / V_armsc_sys_np) * dct_o2_dpp_o2(ppO2_armsc_sys_np)) *
                    q_armsc_sys_in_np * dcb_c_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) -
                dcb_c_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) * q_armsc_sys_in_np *
                    dcb_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)));
    // w.r.t. ppCO2
    wkstiff(64, 64) =
        1. / ts_size +
        theta *
            (-(1. / V_armsc_sys_np) *
                    pow(((dcb_c_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
                             (v_tissmsc_ / V_armsc_sys_np) *
                                 dct_c_o2_dpp_c_o2(ppCO2_armsc_sys_np)) *
                                (dcb_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
                                    (v_tissmsc_ / V_armsc_sys_np) *
                                        dct_o2_dpp_o2(ppO2_armsc_sys_np)) -
                            dcb_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np)),
                        -2.) *
                    ((d2cb_c_o2_dpp_c_o22(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
                         (v_tissmsc_ / V_armsc_sys_np) * d2ct_c_o2_dpp_c_o22(ppCO2_armsc_sys_np)) *
                            (dcb_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
                                (v_tissmsc_ / V_armsc_sys_np) * dct_o2_dpp_o2(ppO2_armsc_sys_np)) +
                        (dcb_c_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
                            (v_tissmsc_ / V_armsc_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_armsc_sys_np)) *
                            d2cb_o2_dpp_o2dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) -
                        d2cb_o2_dpp_c_o22(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) *
                            dcb_c_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) -
                        dcb_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) *
                            d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np)) *
                    ((dcb_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
                         (v_tissmsc_ / V_armsc_sys_np) * dct_o2_dpp_o2(ppO2_armsc_sys_np)) *
                            (q_armsc_sys_in_np * (cb_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) -
                                                     cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                                m_c_o2_armsc_) -
                        dcb_c_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) *
                            (q_armsc_sys_in_np * (cb_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) -
                                                     cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) +
                                m_o2_armsc_ * ct_o2(ppO2_armsc_sys_np) /
                                    (beta + ct_o2(ppO2_armsc_sys_np)))) +
                (1. / V_armsc_sys_np) *
                    pow(((dcb_c_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
                             (v_tissmsc_ / V_armsc_sys_np) *
                                 dct_c_o2_dpp_c_o2(ppCO2_armsc_sys_np)) *
                                (dcb_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
                                    (v_tissmsc_ / V_armsc_sys_np) *
                                        dct_o2_dpp_o2(ppO2_armsc_sys_np)) -
                            dcb_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np)),
                        -1.) *
                    (d2cb_o2_dpp_o2dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) *
                            (q_armsc_sys_in_np * (cb_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) -
                                                     cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                                m_c_o2_armsc_) +
                        (dcb_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
                            (v_tissmsc_ / V_armsc_sys_np) * dct_o2_dpp_o2(ppO2_armsc_sys_np)) *
                            q_armsc_sys_in_np *
                            dcb_c_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) -
                        d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) *
                            (q_armsc_sys_in_np * (cb_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) -
                                                     cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) +
                                m_o2_armsc_ * ct_o2(ppO2_armsc_sys_np) /
                                    (beta + ct_o2(ppO2_armsc_sys_np))) -
                        dcb_c_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) * q_armsc_sys_in_np *
                            dcb_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np)));
    // w.r.t. ppO2
    wkstiff(64, 65) =
        theta *
        (-(1. / V_armsc_sys_np) *
                pow(((dcb_c_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
                         (v_tissmsc_ / V_armsc_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_armsc_sys_np)) *
                            (dcb_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
                                (v_tissmsc_ / V_armsc_sys_np) * dct_o2_dpp_o2(ppO2_armsc_sys_np)) -
                        dcb_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) *
                            dcb_c_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np)),
                    -2.) *
                (d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) *
                        (dcb_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
                            (v_tissmsc_ / V_armsc_sys_np) * dct_o2_dpp_o2(ppO2_armsc_sys_np)) +
                    (dcb_c_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
                        (v_tissmsc_ / V_armsc_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_armsc_sys_np)) *
                        (d2cb_o2_dpp_o22(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
                            (v_tissmsc_ / V_armsc_sys_np) * d2ct_o2_dpp_o22(ppO2_armsc_sys_np)) -
                    d2cb_o2_dpp_o2dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) -
                    dcb_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) *
                        d2cb_c_o2_dpp_o22(ppCO2_armsc_sys_np, ppO2_armsc_sys_np)) *
                ((dcb_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
                     (v_tissmsc_ / V_armsc_sys_np) * dct_o2_dpp_o2(ppO2_armsc_sys_np)) *
                        (q_armsc_sys_in_np * (cb_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) -
                                                 cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                            m_c_o2_armsc_) -
                    dcb_c_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) *
                        (q_armsc_sys_in_np * (cb_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) -
                                                 cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) +
                            m_o2_armsc_ * ct_o2(ppO2_armsc_sys_np) /
                                (beta + ct_o2(ppO2_armsc_sys_np)))) +
            (1. / V_armsc_sys_np) *
                pow(((dcb_c_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
                         (v_tissmsc_ / V_armsc_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_armsc_sys_np)) *
                            (dcb_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
                                (v_tissmsc_ / V_armsc_sys_np) * dct_o2_dpp_o2(ppO2_armsc_sys_np)) -
                        dcb_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) *
                            dcb_c_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np)),
                    -1.) *
                ((d2cb_o2_dpp_o22(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
                     (v_tissmsc_ / V_armsc_sys_np) * d2ct_o2_dpp_o22(ppO2_armsc_sys_np)) *
                        (q_armsc_sys_in_np * (cb_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) -
                                                 cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                            m_c_o2_armsc_) +
                    (dcb_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
                        (v_tissmsc_ / V_armsc_sys_np) * dct_o2_dpp_o2(ppO2_armsc_sys_np)) *
                        q_armsc_sys_in_np * dcb_c_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) -
                    d2cb_c_o2_dpp_o22(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) *
                        (q_armsc_sys_in_np * (cb_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) -
                                                 cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) +
                            m_o2_armsc_ * ct_o2(ppO2_armsc_sys_np) /
                                (beta + ct_o2(ppO2_armsc_sys_np))) -
                    dcb_c_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) *
                        (q_armsc_sys_in_np * dcb_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
                            dct_o2_dpp_o2(ppO2_armsc_sys_np) * beta /
                                ((beta + ct_o2(ppO2_armsc_sys_np)) *
                                    (beta + ct_o2(ppO2_armsc_sys_np))))));

    //////// systemic muscular arteries O2
    // w.r.t. mech. pressure
    wkstiff(65, 6) =
        theta *
        (c_armsc_sys_ * (-1. / (V_armsc_sys_np * V_armsc_sys_np)) *
                pow(((dcb_c_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
                         (v_tissmsc_ / V_armsc_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_armsc_sys_np)) *
                            (dcb_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
                                (v_tissmsc_ / V_armsc_sys_np) * dct_o2_dpp_o2(ppO2_armsc_sys_np)) -
                        dcb_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) *
                            dcb_c_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np)),
                    -1.) *
                ((dcb_c_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
                     (v_tissmsc_ / V_armsc_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_armsc_sys_np)) *
                        (q_armsc_sys_in_np * (cb_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) -
                                                 cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) +
                            m_o2_armsc_ * ct_o2(ppO2_armsc_sys_np) /
                                (beta + ct_o2(ppO2_armsc_sys_np))) -
                    dcb_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) *
                        (q_armsc_sys_in_np * (cb_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) -
                                                 cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                            m_c_o2_armsc_)) +
            (1. / V_armsc_sys_np) *
                (pow(((dcb_c_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
                          (v_tissmsc_ / V_armsc_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_armsc_sys_np)) *
                             (dcb_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
                                 (v_tissmsc_ / V_armsc_sys_np) * dct_o2_dpp_o2(ppO2_armsc_sys_np)) -
                         dcb_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) *
                             dcb_c_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np)),
                    -1.)) *
                (c_armsc_sys_ * (-v_tissmsc_ / (V_armsc_sys_np * V_armsc_sys_np)) *
                    dct_c_o2_dpp_c_o2(ppCO2_armsc_sys_np) *
                    (q_armsc_sys_in_np * (cb_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) -
                                             cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) +
                        m_o2_armsc_ * ct_o2(ppO2_armsc_sys_np) /
                            (beta + ct_o2(ppO2_armsc_sys_np))) *
                    (-pow(((dcb_c_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
                               (v_tissmsc_ / V_armsc_sys_np) *
                                   dct_c_o2_dpp_c_o2(ppCO2_armsc_sys_np)) *
                                  (dcb_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
                                      (v_tissmsc_ / V_armsc_sys_np) *
                                          dct_o2_dpp_o2(ppO2_armsc_sys_np)) -
                              dcb_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) *
                                  dcb_c_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np)),
                         -2.) *
                            (dcb_c_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
                                (v_tissmsc_ / V_armsc_sys_np) *
                                    dct_c_o2_dpp_c_o2(ppCO2_armsc_sys_np)) *
                            c_armsc_sys_ * (-v_tissmsc_ / (V_armsc_sys_np * V_armsc_sys_np)) *
                            dct_o2_dpp_o2(ppO2_armsc_sys_np) +
                        c_armsc_sys_ * (-v_tissmsc_ / (V_armsc_sys_np * V_armsc_sys_np)) *
                            dct_c_o2_dpp_c_o2(ppCO2_armsc_sys_np) *
                            (dcb_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
                                (v_tissmsc_ / V_armsc_sys_np) * dct_o2_dpp_o2(ppO2_armsc_sys_np))) *
                    ((dcb_c_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
                         (v_tissmsc_ / V_armsc_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_armsc_sys_np)) *
                            (q_armsc_sys_in_np * (cb_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) -
                                                     cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) +
                                m_o2_armsc_ * ct_o2(ppO2_armsc_sys_np) /
                                    (beta + ct_o2(ppO2_armsc_sys_np))) -
                        dcb_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) *
                            (q_armsc_sys_in_np * (cb_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) -
                                                     cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                                m_c_o2_armsc_))));
    // w.r.t. upstream flux
    wkstiff(65, 41) =
        theta *
        ((1. / V_armsc_sys_np) *
            pow(((dcb_c_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
                     (v_tissmsc_ / V_armsc_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_armsc_sys_np)) *
                        (dcb_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
                            (v_tissmsc_ / V_armsc_sys_np) * dct_o2_dpp_o2(ppO2_armsc_sys_np)) -
                    dcb_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np)),
                -1.) *
            (dcb_c_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) * 1.0 *
                    (cb_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) -
                        cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                dcb_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) * 1.0 *
                    (cb_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) -
                        cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np))));
    // w.r.t. upstream ppCO2
    wkstiff(65, 58) =
        theta *
        (-(1. / V_armsc_sys_np) *
            pow(((dcb_c_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
                     (v_tissmsc_ / V_armsc_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_armsc_sys_np)) *
                        (dcb_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
                            (v_tissmsc_ / V_armsc_sys_np) * dct_o2_dpp_o2(ppO2_armsc_sys_np)) -
                    dcb_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np)),
                -1.) *
            ((dcb_c_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
                 (v_tissmsc_ / V_armsc_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_armsc_sys_np)) *
                    q_armsc_sys_in_np * dcb_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) -
                dcb_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) * q_armsc_sys_in_np *
                    dcb_c_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)));
    // w.r.t. upstream ppO2
    wkstiff(65, 59) =
        theta *
        (-(1. / V_armsc_sys_np) *
            pow(((dcb_c_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
                     (v_tissmsc_ / V_armsc_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_armsc_sys_np)) *
                        (dcb_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
                            (v_tissmsc_ / V_armsc_sys_np) * dct_o2_dpp_o2(ppO2_armsc_sys_np)) -
                    dcb_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np)),
                -1.) *
            ((dcb_c_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
                 (v_tissmsc_ / V_armsc_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_armsc_sys_np)) *
                    q_armsc_sys_in_np * dcb_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) -
                dcb_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) * q_armsc_sys_in_np *
                    dcb_c_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)));
    // w.r.t. ppCO2
    wkstiff(65, 64) =
        theta *
        (-(1. / V_armsc_sys_np) *
                pow(((dcb_c_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
                         (v_tissmsc_ / V_armsc_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_armsc_sys_np)) *
                            (dcb_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
                                (v_tissmsc_ / V_armsc_sys_np) * dct_o2_dpp_o2(ppO2_armsc_sys_np)) -
                        dcb_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) *
                            dcb_c_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np)),
                    -2.) *
                ((d2cb_c_o2_dpp_c_o22(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
                     (v_tissmsc_ / V_armsc_sys_np) * d2ct_c_o2_dpp_c_o22(ppCO2_armsc_sys_np)) *
                        (dcb_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
                            (v_tissmsc_ / V_armsc_sys_np) * dct_o2_dpp_o2(ppO2_armsc_sys_np)) +
                    (dcb_c_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
                        (v_tissmsc_ / V_armsc_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_armsc_sys_np)) *
                        d2cb_o2_dpp_o2dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) -
                    d2cb_o2_dpp_c_o22(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) -
                    dcb_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) *
                        d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np)) *
                ((dcb_c_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
                     (v_tissmsc_ / V_armsc_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_armsc_sys_np)) *
                        (q_armsc_sys_in_np * (cb_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) -
                                                 cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) +
                            m_o2_armsc_ * ct_o2(ppO2_armsc_sys_np) /
                                (beta + ct_o2(ppO2_armsc_sys_np))) -
                    dcb_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) *
                        (q_armsc_sys_in_np * (cb_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) -
                                                 cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                            m_c_o2_armsc_)) +
            (1. / V_armsc_sys_np) *
                pow(((dcb_c_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
                         (v_tissmsc_ / V_armsc_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_armsc_sys_np)) *
                            (dcb_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
                                (v_tissmsc_ / V_armsc_sys_np) * dct_o2_dpp_o2(ppO2_armsc_sys_np)) -
                        dcb_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) *
                            dcb_c_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np)),
                    -1.) *
                ((d2cb_c_o2_dpp_c_o22(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
                     (v_tissmsc_ / V_armsc_sys_np) * d2ct_c_o2_dpp_c_o22(ppCO2_armsc_sys_np)) *
                        (q_armsc_sys_in_np * (cb_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) -
                                                 cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) +
                            m_o2_armsc_ * ct_o2(ppO2_armsc_sys_np) /
                                (beta + ct_o2(ppO2_armsc_sys_np))) +
                    (dcb_c_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
                        (v_tissmsc_ / V_armsc_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_armsc_sys_np)) *
                        q_armsc_sys_in_np * dcb_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) -
                    d2cb_o2_dpp_c_o22(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) *
                        (q_armsc_sys_in_np * (cb_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) -
                                                 cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                            m_c_o2_armsc_) -
                    dcb_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) * q_armsc_sys_in_np *
                        dcb_c_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np)));
    // w.r.t. ppO2
    wkstiff(65, 65) =
        1. / ts_size +
        theta *
            (-(1. / V_armsc_sys_np) *
                    pow(((dcb_c_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
                             (v_tissmsc_ / V_armsc_sys_np) *
                                 dct_c_o2_dpp_c_o2(ppCO2_armsc_sys_np)) *
                                (dcb_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
                                    (v_tissmsc_ / V_armsc_sys_np) *
                                        dct_o2_dpp_o2(ppO2_armsc_sys_np)) -
                            dcb_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np)),
                        -2.) *
                    (d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) *
                            (dcb_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
                                (v_tissmsc_ / V_armsc_sys_np) * dct_o2_dpp_o2(ppO2_armsc_sys_np)) +
                        (dcb_c_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
                            (v_tissmsc_ / V_armsc_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_armsc_sys_np)) *
                            (d2cb_o2_dpp_o22(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
                                (v_tissmsc_ / V_armsc_sys_np) *
                                    d2ct_o2_dpp_o22(ppO2_armsc_sys_np)) -
                        d2cb_o2_dpp_o2dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) *
                            dcb_c_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) -
                        dcb_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) *
                            d2cb_c_o2_dpp_o22(ppCO2_armsc_sys_np, ppO2_armsc_sys_np)) *
                    ((dcb_c_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
                         (v_tissmsc_ / V_armsc_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_armsc_sys_np)) *
                            (q_armsc_sys_in_np * (cb_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) -
                                                     cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) +
                                m_o2_armsc_ * ct_o2(ppO2_armsc_sys_np) /
                                    (beta + ct_o2(ppO2_armsc_sys_np))) -
                        dcb_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) *
                            (q_armsc_sys_in_np * (cb_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) -
                                                     cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                                m_c_o2_armsc_)) +
                (1. / V_armsc_sys_np) *
                    pow(((dcb_c_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
                             (v_tissmsc_ / V_armsc_sys_np) *
                                 dct_c_o2_dpp_c_o2(ppCO2_armsc_sys_np)) *
                                (dcb_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
                                    (v_tissmsc_ / V_armsc_sys_np) *
                                        dct_o2_dpp_o2(ppO2_armsc_sys_np)) -
                            dcb_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np)),
                        -1.) *
                    (d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) *
                            (q_armsc_sys_in_np * (cb_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) -
                                                     cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) +
                                m_o2_armsc_ * ct_o2(ppO2_armsc_sys_np) /
                                    (beta + ct_o2(ppO2_armsc_sys_np))) +
                        (dcb_c_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
                            (v_tissmsc_ / V_armsc_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_armsc_sys_np)) *
                            (q_armsc_sys_in_np *
                                    dcb_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) +
                                dct_o2_dpp_o2(ppO2_armsc_sys_np) * beta /
                                    ((beta + ct_o2(ppO2_armsc_sys_np)) *
                                        (beta + ct_o2(ppO2_armsc_sys_np)))) -
                        d2cb_o2_dpp_o2dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) *
                            (q_armsc_sys_in_np * (cb_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) -
                                                     cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                                m_c_o2_armsc_) -
                        dcb_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) * q_armsc_sys_in_np *
                            dcb_c_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np)));


    //////// systemic cerebral arteries CO2
    // w.r.t. mech. pressure
    wkstiff(66, 6) =
        theta *
        (c_arcer_sys_ * (-1. / (V_arcer_sys_np * V_arcer_sys_np)) *
                pow(((dcb_c_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
                         (v_tisscer_ / V_arcer_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arcer_sys_np)) *
                            (dcb_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
                                (v_tisscer_ / V_arcer_sys_np) * dct_o2_dpp_o2(ppO2_arcer_sys_np)) -
                        dcb_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) *
                            dcb_c_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np)),
                    -1.) *
                ((dcb_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
                     (v_tisscer_ / V_arcer_sys_np) * dct_o2_dpp_o2(ppO2_arcer_sys_np)) *
                        (q_arcer_sys_in_np * (cb_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) -
                                                 cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                            m_c_o2_arcer_) -
                    dcb_c_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) *
                        (q_arcer_sys_in_np * (cb_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) -
                                                 cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) +
                            m_o2_arcer_ * ct_o2(ppO2_arcer_sys_np) /
                                (beta + ct_o2(ppO2_arcer_sys_np)))) +
            (1. / V_arcer_sys_np) *
                (pow(((dcb_c_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
                          (v_tisscer_ / V_arcer_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arcer_sys_np)) *
                             (dcb_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
                                 (v_tisscer_ / V_arcer_sys_np) * dct_o2_dpp_o2(ppO2_arcer_sys_np)) -
                         dcb_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) *
                             dcb_c_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np)),
                    -1.)) *
                (c_arcer_sys_ * (-v_tisscer_ / (V_arcer_sys_np * V_arcer_sys_np)) *
                    dct_o2_dpp_o2(ppO2_arcer_sys_np) *
                    (q_arcer_sys_in_np * (cb_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) -
                                             cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                        m_c_o2_arcer_) *
                    (-pow(((dcb_c_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
                               (v_tisscer_ / V_arcer_sys_np) *
                                   dct_c_o2_dpp_c_o2(ppCO2_arcer_sys_np)) *
                                  (dcb_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
                                      (v_tisscer_ / V_arcer_sys_np) *
                                          dct_o2_dpp_o2(ppO2_arcer_sys_np)) -
                              dcb_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) *
                                  dcb_c_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np)),
                         -2.) *
                            (dcb_c_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
                                (v_tisscer_ / V_arcer_sys_np) *
                                    dct_c_o2_dpp_c_o2(ppCO2_arcer_sys_np)) *
                            c_arcer_sys_ * (-v_tisscer_ / (V_arcer_sys_np * V_arcer_sys_np)) *
                            dct_o2_dpp_o2(ppO2_arcer_sys_np) +
                        c_arcer_sys_ * (-v_tisscer_ / (V_arcer_sys_np * V_arcer_sys_np)) *
                            dct_c_o2_dpp_c_o2(ppCO2_arcer_sys_np) *
                            (dcb_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
                                (v_tisscer_ / V_arcer_sys_np) * dct_o2_dpp_o2(ppO2_arcer_sys_np))) *
                    ((dcb_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
                         (v_tisscer_ / V_arcer_sys_np) * dct_o2_dpp_o2(ppO2_arcer_sys_np)) *
                            (q_arcer_sys_in_np * (cb_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) -
                                                     cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                                m_c_o2_arcer_) -
                        dcb_c_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) *
                            (q_arcer_sys_in_np * (cb_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) -
                                                     cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) +
                                m_o2_arcer_ * ct_o2(ppO2_arcer_sys_np) /
                                    (beta + ct_o2(ppO2_arcer_sys_np))))));
    // w.r.t. upstream flux
    wkstiff(66, 42) =
        theta *
        ((1. / V_arcer_sys_np) *
            pow(((dcb_c_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
                     (v_tisscer_ / V_arcer_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arcer_sys_np)) *
                        (dcb_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
                            (v_tisscer_ / V_arcer_sys_np) * dct_o2_dpp_o2(ppO2_arcer_sys_np)) -
                    dcb_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np)),
                -1.) *
            (dcb_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) * 1.0 *
                    (cb_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) -
                        cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                dcb_c_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) * 1.0 *
                    (cb_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) -
                        cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np))));
    // w.r.t. upstream ppCO2
    wkstiff(66, 58) =
        theta *
        (-(1. / V_arcer_sys_np) *
            pow(((dcb_c_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
                     (v_tisscer_ / V_arcer_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arcer_sys_np)) *
                        (dcb_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
                            (v_tisscer_ / V_arcer_sys_np) * dct_o2_dpp_o2(ppO2_arcer_sys_np)) -
                    dcb_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np)),
                -1.) *
            ((dcb_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
                 (v_tisscer_ / V_arcer_sys_np) * dct_o2_dpp_o2(ppO2_arcer_sys_np)) *
                    q_arcer_sys_in_np * dcb_c_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) -
                dcb_c_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) * q_arcer_sys_in_np *
                    dcb_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)));
    // w.r.t. upstream ppO2
    wkstiff(66, 59) =
        theta *
        (-(1. / V_arcer_sys_np) *
            pow(((dcb_c_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
                     (v_tisscer_ / V_arcer_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arcer_sys_np)) *
                        (dcb_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
                            (v_tisscer_ / V_arcer_sys_np) * dct_o2_dpp_o2(ppO2_arcer_sys_np)) -
                    dcb_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np)),
                -1.) *
            ((dcb_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
                 (v_tisscer_ / V_arcer_sys_np) * dct_o2_dpp_o2(ppO2_arcer_sys_np)) *
                    q_arcer_sys_in_np * dcb_c_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) -
                dcb_c_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) * q_arcer_sys_in_np *
                    dcb_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)));
    // w.r.t. ppCO2
    wkstiff(66, 66) =
        1. / ts_size +
        theta *
            (-(1. / V_arcer_sys_np) *
                    pow(((dcb_c_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
                             (v_tisscer_ / V_arcer_sys_np) *
                                 dct_c_o2_dpp_c_o2(ppCO2_arcer_sys_np)) *
                                (dcb_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
                                    (v_tisscer_ / V_arcer_sys_np) *
                                        dct_o2_dpp_o2(ppO2_arcer_sys_np)) -
                            dcb_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np)),
                        -2.) *
                    ((d2cb_c_o2_dpp_c_o22(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
                         (v_tisscer_ / V_arcer_sys_np) * d2ct_c_o2_dpp_c_o22(ppCO2_arcer_sys_np)) *
                            (dcb_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
                                (v_tisscer_ / V_arcer_sys_np) * dct_o2_dpp_o2(ppO2_arcer_sys_np)) +
                        (dcb_c_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
                            (v_tisscer_ / V_arcer_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arcer_sys_np)) *
                            d2cb_o2_dpp_o2dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) -
                        d2cb_o2_dpp_c_o22(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) *
                            dcb_c_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) -
                        dcb_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) *
                            d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np)) *
                    ((dcb_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
                         (v_tisscer_ / V_arcer_sys_np) * dct_o2_dpp_o2(ppO2_arcer_sys_np)) *
                            (q_arcer_sys_in_np * (cb_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) -
                                                     cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                                m_c_o2_arcer_) -
                        dcb_c_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) *
                            (q_arcer_sys_in_np * (cb_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) -
                                                     cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) +
                                m_o2_arcer_ * ct_o2(ppO2_arcer_sys_np) /
                                    (beta + ct_o2(ppO2_arcer_sys_np)))) +
                (1. / V_arcer_sys_np) *
                    pow(((dcb_c_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
                             (v_tisscer_ / V_arcer_sys_np) *
                                 dct_c_o2_dpp_c_o2(ppCO2_arcer_sys_np)) *
                                (dcb_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
                                    (v_tisscer_ / V_arcer_sys_np) *
                                        dct_o2_dpp_o2(ppO2_arcer_sys_np)) -
                            dcb_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np)),
                        -1.) *
                    (d2cb_o2_dpp_o2dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) *
                            (q_arcer_sys_in_np * (cb_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) -
                                                     cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                                m_c_o2_arcer_) +
                        (dcb_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
                            (v_tisscer_ / V_arcer_sys_np) * dct_o2_dpp_o2(ppO2_arcer_sys_np)) *
                            q_arcer_sys_in_np *
                            dcb_c_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) -
                        d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) *
                            (q_arcer_sys_in_np * (cb_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) -
                                                     cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) +
                                m_o2_arcer_ * ct_o2(ppO2_arcer_sys_np) /
                                    (beta + ct_o2(ppO2_arcer_sys_np))) -
                        dcb_c_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) * q_arcer_sys_in_np *
                            dcb_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np)));
    // w.r.t. ppO2
    wkstiff(66, 67) =
        theta *
        (-(1. / V_arcer_sys_np) *
                pow(((dcb_c_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
                         (v_tisscer_ / V_arcer_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arcer_sys_np)) *
                            (dcb_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
                                (v_tisscer_ / V_arcer_sys_np) * dct_o2_dpp_o2(ppO2_arcer_sys_np)) -
                        dcb_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) *
                            dcb_c_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np)),
                    -2.) *
                (d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) *
                        (dcb_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
                            (v_tisscer_ / V_arcer_sys_np) * dct_o2_dpp_o2(ppO2_arcer_sys_np)) +
                    (dcb_c_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
                        (v_tisscer_ / V_arcer_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arcer_sys_np)) *
                        (d2cb_o2_dpp_o22(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
                            (v_tisscer_ / V_arcer_sys_np) * d2ct_o2_dpp_o22(ppO2_arcer_sys_np)) -
                    d2cb_o2_dpp_o2dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) -
                    dcb_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) *
                        d2cb_c_o2_dpp_o22(ppCO2_arcer_sys_np, ppO2_arcer_sys_np)) *
                ((dcb_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
                     (v_tisscer_ / V_arcer_sys_np) * dct_o2_dpp_o2(ppO2_arcer_sys_np)) *
                        (q_arcer_sys_in_np * (cb_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) -
                                                 cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                            m_c_o2_arcer_) -
                    dcb_c_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) *
                        (q_arcer_sys_in_np * (cb_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) -
                                                 cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) +
                            m_o2_arcer_ * ct_o2(ppO2_arcer_sys_np) /
                                (beta + ct_o2(ppO2_arcer_sys_np)))) +
            (1. / V_arcer_sys_np) *
                pow(((dcb_c_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
                         (v_tisscer_ / V_arcer_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arcer_sys_np)) *
                            (dcb_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
                                (v_tisscer_ / V_arcer_sys_np) * dct_o2_dpp_o2(ppO2_arcer_sys_np)) -
                        dcb_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) *
                            dcb_c_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np)),
                    -1.) *
                ((d2cb_o2_dpp_o22(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
                     (v_tisscer_ / V_arcer_sys_np) * d2ct_o2_dpp_o22(ppO2_arcer_sys_np)) *
                        (q_arcer_sys_in_np * (cb_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) -
                                                 cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                            m_c_o2_arcer_) +
                    (dcb_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
                        (v_tisscer_ / V_arcer_sys_np) * dct_o2_dpp_o2(ppO2_arcer_sys_np)) *
                        q_arcer_sys_in_np * dcb_c_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) -
                    d2cb_c_o2_dpp_o22(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) *
                        (q_arcer_sys_in_np * (cb_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) -
                                                 cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) +
                            m_o2_arcer_ * ct_o2(ppO2_arcer_sys_np) /
                                (beta + ct_o2(ppO2_arcer_sys_np))) -
                    dcb_c_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) *
                        (q_arcer_sys_in_np * dcb_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
                            dct_o2_dpp_o2(ppO2_arcer_sys_np) * beta /
                                ((beta + ct_o2(ppO2_arcer_sys_np)) *
                                    (beta + ct_o2(ppO2_arcer_sys_np))))));

    //////// systemic cerebral arteries O2
    // w.r.t. mech. pressure
    wkstiff(67, 6) =
        theta *
        (c_arcer_sys_ * (-1. / (V_arcer_sys_np * V_arcer_sys_np)) *
                pow(((dcb_c_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
                         (v_tisscer_ / V_arcer_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arcer_sys_np)) *
                            (dcb_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
                                (v_tisscer_ / V_arcer_sys_np) * dct_o2_dpp_o2(ppO2_arcer_sys_np)) -
                        dcb_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) *
                            dcb_c_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np)),
                    -1.) *
                ((dcb_c_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
                     (v_tisscer_ / V_arcer_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arcer_sys_np)) *
                        (q_arcer_sys_in_np * (cb_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) -
                                                 cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) +
                            m_o2_arcer_ * ct_o2(ppO2_arcer_sys_np) /
                                (beta + ct_o2(ppO2_arcer_sys_np))) -
                    dcb_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) *
                        (q_arcer_sys_in_np * (cb_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) -
                                                 cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                            m_c_o2_arcer_)) +
            (1. / V_arcer_sys_np) *
                (pow(((dcb_c_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
                          (v_tisscer_ / V_arcer_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arcer_sys_np)) *
                             (dcb_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
                                 (v_tisscer_ / V_arcer_sys_np) * dct_o2_dpp_o2(ppO2_arcer_sys_np)) -
                         dcb_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) *
                             dcb_c_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np)),
                    -1.)) *
                (c_arcer_sys_ * (-v_tisscer_ / (V_arcer_sys_np * V_arcer_sys_np)) *
                    dct_c_o2_dpp_c_o2(ppCO2_arcer_sys_np) *
                    (q_arcer_sys_in_np * (cb_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) -
                                             cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) +
                        m_o2_arcer_ * ct_o2(ppO2_arcer_sys_np) /
                            (beta + ct_o2(ppO2_arcer_sys_np))) *
                    (-pow(((dcb_c_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
                               (v_tisscer_ / V_arcer_sys_np) *
                                   dct_c_o2_dpp_c_o2(ppCO2_arcer_sys_np)) *
                                  (dcb_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
                                      (v_tisscer_ / V_arcer_sys_np) *
                                          dct_o2_dpp_o2(ppO2_arcer_sys_np)) -
                              dcb_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) *
                                  dcb_c_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np)),
                         -2.) *
                            (dcb_c_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
                                (v_tisscer_ / V_arcer_sys_np) *
                                    dct_c_o2_dpp_c_o2(ppCO2_arcer_sys_np)) *
                            c_arcer_sys_ * (-v_tisscer_ / (V_arcer_sys_np * V_arcer_sys_np)) *
                            dct_o2_dpp_o2(ppO2_arcer_sys_np) +
                        c_arcer_sys_ * (-v_tisscer_ / (V_arcer_sys_np * V_arcer_sys_np)) *
                            dct_c_o2_dpp_c_o2(ppCO2_arcer_sys_np) *
                            (dcb_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
                                (v_tisscer_ / V_arcer_sys_np) * dct_o2_dpp_o2(ppO2_arcer_sys_np))) *
                    ((dcb_c_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
                         (v_tisscer_ / V_arcer_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arcer_sys_np)) *
                            (q_arcer_sys_in_np * (cb_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) -
                                                     cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) +
                                m_o2_arcer_ * ct_o2(ppO2_arcer_sys_np) /
                                    (beta + ct_o2(ppO2_arcer_sys_np))) -
                        dcb_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) *
                            (q_arcer_sys_in_np * (cb_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) -
                                                     cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                                m_c_o2_arcer_))));
    // w.r.t. upstream flux
    wkstiff(67, 42) =
        theta *
        ((1. / V_arcer_sys_np) *
            pow(((dcb_c_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
                     (v_tisscer_ / V_arcer_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arcer_sys_np)) *
                        (dcb_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
                            (v_tisscer_ / V_arcer_sys_np) * dct_o2_dpp_o2(ppO2_arcer_sys_np)) -
                    dcb_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np)),
                -1.) *
            (dcb_c_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) * 1.0 *
                    (cb_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) -
                        cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                dcb_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) * 1.0 *
                    (cb_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) -
                        cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np))));
    // w.r.t. upstream ppCO2
    wkstiff(67, 58) =
        theta *
        (-(1. / V_arcer_sys_np) *
            pow(((dcb_c_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
                     (v_tisscer_ / V_arcer_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arcer_sys_np)) *
                        (dcb_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
                            (v_tisscer_ / V_arcer_sys_np) * dct_o2_dpp_o2(ppO2_arcer_sys_np)) -
                    dcb_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np)),
                -1.) *
            ((dcb_c_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
                 (v_tisscer_ / V_arcer_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arcer_sys_np)) *
                    q_arcer_sys_in_np * dcb_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) -
                dcb_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) * q_arcer_sys_in_np *
                    dcb_c_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)));
    // w.r.t. upstream ppO2
    wkstiff(67, 59) =
        theta *
        (-(1. / V_arcer_sys_np) *
            pow(((dcb_c_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
                     (v_tisscer_ / V_arcer_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arcer_sys_np)) *
                        (dcb_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
                            (v_tisscer_ / V_arcer_sys_np) * dct_o2_dpp_o2(ppO2_arcer_sys_np)) -
                    dcb_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np)),
                -1.) *
            ((dcb_c_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
                 (v_tisscer_ / V_arcer_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arcer_sys_np)) *
                    q_arcer_sys_in_np * dcb_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) -
                dcb_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) * q_arcer_sys_in_np *
                    dcb_c_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)));
    // w.r.t. ppCO2
    wkstiff(67, 66) =
        theta *
        (-(1. / V_arcer_sys_np) *
                pow(((dcb_c_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
                         (v_tisscer_ / V_arcer_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arcer_sys_np)) *
                            (dcb_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
                                (v_tisscer_ / V_arcer_sys_np) * dct_o2_dpp_o2(ppO2_arcer_sys_np)) -
                        dcb_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) *
                            dcb_c_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np)),
                    -2.) *
                ((d2cb_c_o2_dpp_c_o22(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
                     (v_tisscer_ / V_arcer_sys_np) * d2ct_c_o2_dpp_c_o22(ppCO2_arcer_sys_np)) *
                        (dcb_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
                            (v_tisscer_ / V_arcer_sys_np) * dct_o2_dpp_o2(ppO2_arcer_sys_np)) +
                    (dcb_c_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
                        (v_tisscer_ / V_arcer_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arcer_sys_np)) *
                        d2cb_o2_dpp_o2dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) -
                    d2cb_o2_dpp_c_o22(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) -
                    dcb_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) *
                        d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np)) *
                ((dcb_c_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
                     (v_tisscer_ / V_arcer_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arcer_sys_np)) *
                        (q_arcer_sys_in_np * (cb_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) -
                                                 cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) +
                            m_o2_arcer_ * ct_o2(ppO2_arcer_sys_np) /
                                (beta + ct_o2(ppO2_arcer_sys_np))) -
                    dcb_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) *
                        (q_arcer_sys_in_np * (cb_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) -
                                                 cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                            m_c_o2_arcer_)) +
            (1. / V_arcer_sys_np) *
                pow(((dcb_c_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
                         (v_tisscer_ / V_arcer_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arcer_sys_np)) *
                            (dcb_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
                                (v_tisscer_ / V_arcer_sys_np) * dct_o2_dpp_o2(ppO2_arcer_sys_np)) -
                        dcb_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) *
                            dcb_c_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np)),
                    -1.) *
                ((d2cb_c_o2_dpp_c_o22(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
                     (v_tisscer_ / V_arcer_sys_np) * d2ct_c_o2_dpp_c_o22(ppCO2_arcer_sys_np)) *
                        (q_arcer_sys_in_np * (cb_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) -
                                                 cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) +
                            m_o2_arcer_ * ct_o2(ppO2_arcer_sys_np) /
                                (beta + ct_o2(ppO2_arcer_sys_np))) +
                    (dcb_c_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
                        (v_tisscer_ / V_arcer_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arcer_sys_np)) *
                        q_arcer_sys_in_np * dcb_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) -
                    d2cb_o2_dpp_c_o22(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) *
                        (q_arcer_sys_in_np * (cb_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) -
                                                 cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                            m_c_o2_arcer_) -
                    dcb_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) * q_arcer_sys_in_np *
                        dcb_c_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np)));
    // w.r.t. ppO2
    wkstiff(67, 67) =
        1. / ts_size +
        theta *
            (-(1. / V_arcer_sys_np) *
                    pow(((dcb_c_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
                             (v_tisscer_ / V_arcer_sys_np) *
                                 dct_c_o2_dpp_c_o2(ppCO2_arcer_sys_np)) *
                                (dcb_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
                                    (v_tisscer_ / V_arcer_sys_np) *
                                        dct_o2_dpp_o2(ppO2_arcer_sys_np)) -
                            dcb_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np)),
                        -2.) *
                    (d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) *
                            (dcb_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
                                (v_tisscer_ / V_arcer_sys_np) * dct_o2_dpp_o2(ppO2_arcer_sys_np)) +
                        (dcb_c_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
                            (v_tisscer_ / V_arcer_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arcer_sys_np)) *
                            (d2cb_o2_dpp_o22(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
                                (v_tisscer_ / V_arcer_sys_np) *
                                    d2ct_o2_dpp_o22(ppO2_arcer_sys_np)) -
                        d2cb_o2_dpp_o2dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) *
                            dcb_c_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) -
                        dcb_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) *
                            d2cb_c_o2_dpp_o22(ppCO2_arcer_sys_np, ppO2_arcer_sys_np)) *
                    ((dcb_c_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
                         (v_tisscer_ / V_arcer_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arcer_sys_np)) *
                            (q_arcer_sys_in_np * (cb_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) -
                                                     cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) +
                                m_o2_arcer_ * ct_o2(ppO2_arcer_sys_np) /
                                    (beta + ct_o2(ppO2_arcer_sys_np))) -
                        dcb_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) *
                            (q_arcer_sys_in_np * (cb_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) -
                                                     cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                                m_c_o2_arcer_)) +
                (1. / V_arcer_sys_np) *
                    pow(((dcb_c_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
                             (v_tisscer_ / V_arcer_sys_np) *
                                 dct_c_o2_dpp_c_o2(ppCO2_arcer_sys_np)) *
                                (dcb_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
                                    (v_tisscer_ / V_arcer_sys_np) *
                                        dct_o2_dpp_o2(ppO2_arcer_sys_np)) -
                            dcb_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np)),
                        -1.) *
                    (d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) *
                            (q_arcer_sys_in_np * (cb_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) -
                                                     cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) +
                                m_o2_arcer_ * ct_o2(ppO2_arcer_sys_np) /
                                    (beta + ct_o2(ppO2_arcer_sys_np))) +
                        (dcb_c_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
                            (v_tisscer_ / V_arcer_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arcer_sys_np)) *
                            (q_arcer_sys_in_np *
                                    dcb_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) +
                                dct_o2_dpp_o2(ppO2_arcer_sys_np) * beta /
                                    ((beta + ct_o2(ppO2_arcer_sys_np)) *
                                        (beta + ct_o2(ppO2_arcer_sys_np)))) -
                        d2cb_o2_dpp_o2dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) *
                            (q_arcer_sys_in_np * (cb_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) -
                                                     cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                                m_c_o2_arcer_) -
                        dcb_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) * q_arcer_sys_in_np *
                            dcb_c_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np)));


    //////// systemic coronary arteries CO2
    // w.r.t. mech. pressure
    wkstiff(68, 6) =
        theta *
        (c_arcor_sys_ * (-1. / (V_arcor_sys_np * V_arcor_sys_np)) *
                pow(((dcb_c_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
                         (v_tisscor_ / V_arcor_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arcor_sys_np)) *
                            (dcb_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
                                (v_tisscor_ / V_arcor_sys_np) * dct_o2_dpp_o2(ppO2_arcor_sys_np)) -
                        dcb_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) *
                            dcb_c_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np)),
                    -1.) *
                ((dcb_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
                     (v_tisscor_ / V_arcor_sys_np) * dct_o2_dpp_o2(ppO2_arcor_sys_np)) *
                        (q_arcor_sys_in_np * (cb_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) -
                                                 cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                            m_c_o2_arcor_) -
                    dcb_c_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) *
                        (q_arcor_sys_in_np * (cb_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) -
                                                 cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) +
                            m_o2_arcor_ * ct_o2(ppO2_arcor_sys_np) /
                                (beta + ct_o2(ppO2_arcor_sys_np)))) +
            (1. / V_arcor_sys_np) *
                (pow(((dcb_c_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
                          (v_tisscor_ / V_arcor_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arcor_sys_np)) *
                             (dcb_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
                                 (v_tisscor_ / V_arcor_sys_np) * dct_o2_dpp_o2(ppO2_arcor_sys_np)) -
                         dcb_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) *
                             dcb_c_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np)),
                    -1.)) *
                (c_arcor_sys_ * (-v_tisscor_ / (V_arcor_sys_np * V_arcor_sys_np)) *
                    dct_o2_dpp_o2(ppO2_arcor_sys_np) *
                    (q_arcor_sys_in_np * (cb_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) -
                                             cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                        m_c_o2_arcor_) *
                    (-pow(((dcb_c_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
                               (v_tisscor_ / V_arcor_sys_np) *
                                   dct_c_o2_dpp_c_o2(ppCO2_arcor_sys_np)) *
                                  (dcb_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
                                      (v_tisscor_ / V_arcor_sys_np) *
                                          dct_o2_dpp_o2(ppO2_arcor_sys_np)) -
                              dcb_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) *
                                  dcb_c_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np)),
                         -2.) *
                            (dcb_c_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
                                (v_tisscor_ / V_arcor_sys_np) *
                                    dct_c_o2_dpp_c_o2(ppCO2_arcor_sys_np)) *
                            c_arcor_sys_ * (-v_tisscor_ / (V_arcor_sys_np * V_arcor_sys_np)) *
                            dct_o2_dpp_o2(ppO2_arcor_sys_np) +
                        c_arcor_sys_ * (-v_tisscor_ / (V_arcor_sys_np * V_arcor_sys_np)) *
                            dct_c_o2_dpp_c_o2(ppCO2_arcor_sys_np) *
                            (dcb_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
                                (v_tisscor_ / V_arcor_sys_np) * dct_o2_dpp_o2(ppO2_arcor_sys_np))) *
                    ((dcb_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
                         (v_tisscor_ / V_arcor_sys_np) * dct_o2_dpp_o2(ppO2_arcor_sys_np)) *
                            (q_arcor_sys_in_np * (cb_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) -
                                                     cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                                m_c_o2_arcor_) -
                        dcb_c_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) *
                            (q_arcor_sys_in_np * (cb_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) -
                                                     cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) +
                                m_o2_arcor_ * ct_o2(ppO2_arcor_sys_np) /
                                    (beta + ct_o2(ppO2_arcor_sys_np))))));
    // w.r.t. upstream flux
    wkstiff(68, 43) =
        theta *
        ((1. / V_arcor_sys_np) *
            pow(((dcb_c_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
                     (v_tisscor_ / V_arcor_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arcor_sys_np)) *
                        (dcb_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
                            (v_tisscor_ / V_arcor_sys_np) * dct_o2_dpp_o2(ppO2_arcor_sys_np)) -
                    dcb_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np)),
                -1.) *
            (dcb_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) * 1.0 *
                    (cb_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) -
                        cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                dcb_c_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) * 1.0 *
                    (cb_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) -
                        cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np))));
    // w.r.t. upstream ppCO2
    wkstiff(68, 58) =
        theta *
        (-(1. / V_arcor_sys_np) *
            pow(((dcb_c_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
                     (v_tisscor_ / V_arcor_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arcor_sys_np)) *
                        (dcb_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
                            (v_tisscor_ / V_arcor_sys_np) * dct_o2_dpp_o2(ppO2_arcor_sys_np)) -
                    dcb_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np)),
                -1.) *
            ((dcb_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
                 (v_tisscor_ / V_arcor_sys_np) * dct_o2_dpp_o2(ppO2_arcor_sys_np)) *
                    q_arcor_sys_in_np * dcb_c_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) -
                dcb_c_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) * q_arcor_sys_in_np *
                    dcb_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)));
    // w.r.t. upstream ppO2
    wkstiff(68, 59) =
        theta *
        (-(1. / V_arcor_sys_np) *
            pow(((dcb_c_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
                     (v_tisscor_ / V_arcor_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arcor_sys_np)) *
                        (dcb_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
                            (v_tisscor_ / V_arcor_sys_np) * dct_o2_dpp_o2(ppO2_arcor_sys_np)) -
                    dcb_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np)),
                -1.) *
            ((dcb_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
                 (v_tisscor_ / V_arcor_sys_np) * dct_o2_dpp_o2(ppO2_arcor_sys_np)) *
                    q_arcor_sys_in_np * dcb_c_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) -
                dcb_c_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) * q_arcor_sys_in_np *
                    dcb_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)));
    // w.r.t. ppCO2
    wkstiff(68, 68) =
        1. / ts_size +
        theta *
            (-(1. / V_arcor_sys_np) *
                    pow(((dcb_c_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
                             (v_tisscor_ / V_arcor_sys_np) *
                                 dct_c_o2_dpp_c_o2(ppCO2_arcor_sys_np)) *
                                (dcb_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
                                    (v_tisscor_ / V_arcor_sys_np) *
                                        dct_o2_dpp_o2(ppO2_arcor_sys_np)) -
                            dcb_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np)),
                        -2.) *
                    ((d2cb_c_o2_dpp_c_o22(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
                         (v_tisscor_ / V_arcor_sys_np) * d2ct_c_o2_dpp_c_o22(ppCO2_arcor_sys_np)) *
                            (dcb_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
                                (v_tisscor_ / V_arcor_sys_np) * dct_o2_dpp_o2(ppO2_arcor_sys_np)) +
                        (dcb_c_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
                            (v_tisscor_ / V_arcor_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arcor_sys_np)) *
                            d2cb_o2_dpp_o2dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) -
                        d2cb_o2_dpp_c_o22(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) *
                            dcb_c_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) -
                        dcb_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) *
                            d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np)) *
                    ((dcb_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
                         (v_tisscor_ / V_arcor_sys_np) * dct_o2_dpp_o2(ppO2_arcor_sys_np)) *
                            (q_arcor_sys_in_np * (cb_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) -
                                                     cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                                m_c_o2_arcor_) -
                        dcb_c_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) *
                            (q_arcor_sys_in_np * (cb_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) -
                                                     cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) +
                                m_o2_arcor_ * ct_o2(ppO2_arcor_sys_np) /
                                    (beta + ct_o2(ppO2_arcor_sys_np)))) +
                (1. / V_arcor_sys_np) *
                    pow(((dcb_c_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
                             (v_tisscor_ / V_arcor_sys_np) *
                                 dct_c_o2_dpp_c_o2(ppCO2_arcor_sys_np)) *
                                (dcb_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
                                    (v_tisscor_ / V_arcor_sys_np) *
                                        dct_o2_dpp_o2(ppO2_arcor_sys_np)) -
                            dcb_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np)),
                        -1.) *
                    (d2cb_o2_dpp_o2dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) *
                            (q_arcor_sys_in_np * (cb_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) -
                                                     cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                                m_c_o2_arcor_) +
                        (dcb_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
                            (v_tisscor_ / V_arcor_sys_np) * dct_o2_dpp_o2(ppO2_arcor_sys_np)) *
                            q_arcor_sys_in_np *
                            dcb_c_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) -
                        d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) *
                            (q_arcor_sys_in_np * (cb_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) -
                                                     cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) +
                                m_o2_arcor_ * ct_o2(ppO2_arcor_sys_np) /
                                    (beta + ct_o2(ppO2_arcor_sys_np))) -
                        dcb_c_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) * q_arcor_sys_in_np *
                            dcb_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np)));
    // w.r.t. ppO2
    wkstiff(68, 69) =
        theta *
        (-(1. / V_arcor_sys_np) *
                pow(((dcb_c_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
                         (v_tisscor_ / V_arcor_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arcor_sys_np)) *
                            (dcb_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
                                (v_tisscor_ / V_arcor_sys_np) * dct_o2_dpp_o2(ppO2_arcor_sys_np)) -
                        dcb_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) *
                            dcb_c_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np)),
                    -2.) *
                (d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) *
                        (dcb_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
                            (v_tisscor_ / V_arcor_sys_np) * dct_o2_dpp_o2(ppO2_arcor_sys_np)) +
                    (dcb_c_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
                        (v_tisscor_ / V_arcor_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arcor_sys_np)) *
                        (d2cb_o2_dpp_o22(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
                            (v_tisscor_ / V_arcor_sys_np) * d2ct_o2_dpp_o22(ppO2_arcor_sys_np)) -
                    d2cb_o2_dpp_o2dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) -
                    dcb_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) *
                        d2cb_c_o2_dpp_o22(ppCO2_arcor_sys_np, ppO2_arcor_sys_np)) *
                ((dcb_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
                     (v_tisscor_ / V_arcor_sys_np) * dct_o2_dpp_o2(ppO2_arcor_sys_np)) *
                        (q_arcor_sys_in_np * (cb_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) -
                                                 cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                            m_c_o2_arcor_) -
                    dcb_c_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) *
                        (q_arcor_sys_in_np * (cb_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) -
                                                 cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) +
                            m_o2_arcor_ * ct_o2(ppO2_arcor_sys_np) /
                                (beta + ct_o2(ppO2_arcor_sys_np)))) +
            (1. / V_arcor_sys_np) *
                pow(((dcb_c_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
                         (v_tisscor_ / V_arcor_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arcor_sys_np)) *
                            (dcb_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
                                (v_tisscor_ / V_arcor_sys_np) * dct_o2_dpp_o2(ppO2_arcor_sys_np)) -
                        dcb_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) *
                            dcb_c_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np)),
                    -1.) *
                ((d2cb_o2_dpp_o22(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
                     (v_tisscor_ / V_arcor_sys_np) * d2ct_o2_dpp_o22(ppO2_arcor_sys_np)) *
                        (q_arcor_sys_in_np * (cb_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) -
                                                 cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                            m_c_o2_arcor_) +
                    (dcb_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
                        (v_tisscor_ / V_arcor_sys_np) * dct_o2_dpp_o2(ppO2_arcor_sys_np)) *
                        q_arcor_sys_in_np * dcb_c_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) -
                    d2cb_c_o2_dpp_o22(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) *
                        (q_arcor_sys_in_np * (cb_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) -
                                                 cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) +
                            m_o2_arcor_ * ct_o2(ppO2_arcor_sys_np) /
                                (beta + ct_o2(ppO2_arcor_sys_np))) -
                    dcb_c_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) *
                        (q_arcor_sys_in_np * dcb_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
                            dct_o2_dpp_o2(ppO2_arcor_sys_np) * beta /
                                ((beta + ct_o2(ppO2_arcor_sys_np)) *
                                    (beta + ct_o2(ppO2_arcor_sys_np))))));

    //////// systemic coronary arteries O2
    // w.r.t. mech. pressure
    wkstiff(69, 6) =
        theta *
        (c_arcor_sys_ * (-1. / (V_arcor_sys_np * V_arcor_sys_np)) *
                pow(((dcb_c_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
                         (v_tisscor_ / V_arcor_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arcor_sys_np)) *
                            (dcb_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
                                (v_tisscor_ / V_arcor_sys_np) * dct_o2_dpp_o2(ppO2_arcor_sys_np)) -
                        dcb_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) *
                            dcb_c_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np)),
                    -1.) *
                ((dcb_c_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
                     (v_tisscor_ / V_arcor_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arcor_sys_np)) *
                        (q_arcor_sys_in_np * (cb_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) -
                                                 cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) +
                            m_o2_arcor_ * ct_o2(ppO2_arcor_sys_np) /
                                (beta + ct_o2(ppO2_arcor_sys_np))) -
                    dcb_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) *
                        (q_arcor_sys_in_np * (cb_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) -
                                                 cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                            m_c_o2_arcor_)) +
            (1. / V_arcor_sys_np) *
                (pow(((dcb_c_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
                          (v_tisscor_ / V_arcor_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arcor_sys_np)) *
                             (dcb_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
                                 (v_tisscor_ / V_arcor_sys_np) * dct_o2_dpp_o2(ppO2_arcor_sys_np)) -
                         dcb_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) *
                             dcb_c_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np)),
                    -1.)) *
                (c_arcor_sys_ * (-v_tisscor_ / (V_arcor_sys_np * V_arcor_sys_np)) *
                    dct_c_o2_dpp_c_o2(ppCO2_arcor_sys_np) *
                    (q_arcor_sys_in_np * (cb_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) -
                                             cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) +
                        m_o2_arcor_ * ct_o2(ppO2_arcor_sys_np) /
                            (beta + ct_o2(ppO2_arcor_sys_np))) *
                    (-pow(((dcb_c_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
                               (v_tisscor_ / V_arcor_sys_np) *
                                   dct_c_o2_dpp_c_o2(ppCO2_arcor_sys_np)) *
                                  (dcb_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
                                      (v_tisscor_ / V_arcor_sys_np) *
                                          dct_o2_dpp_o2(ppO2_arcor_sys_np)) -
                              dcb_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) *
                                  dcb_c_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np)),
                         -2.) *
                            (dcb_c_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
                                (v_tisscor_ / V_arcor_sys_np) *
                                    dct_c_o2_dpp_c_o2(ppCO2_arcor_sys_np)) *
                            c_arcor_sys_ * (-v_tisscor_ / (V_arcor_sys_np * V_arcor_sys_np)) *
                            dct_o2_dpp_o2(ppO2_arcor_sys_np) +
                        c_arcor_sys_ * (-v_tisscor_ / (V_arcor_sys_np * V_arcor_sys_np)) *
                            dct_c_o2_dpp_c_o2(ppCO2_arcor_sys_np) *
                            (dcb_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
                                (v_tisscor_ / V_arcor_sys_np) * dct_o2_dpp_o2(ppO2_arcor_sys_np))) *
                    ((dcb_c_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
                         (v_tisscor_ / V_arcor_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arcor_sys_np)) *
                            (q_arcor_sys_in_np * (cb_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) -
                                                     cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) +
                                m_o2_arcor_ * ct_o2(ppO2_arcor_sys_np) /
                                    (beta + ct_o2(ppO2_arcor_sys_np))) -
                        dcb_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) *
                            (q_arcor_sys_in_np * (cb_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) -
                                                     cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                                m_c_o2_arcor_))));
    // w.r.t. upstream flux
    wkstiff(69, 43) =
        theta *
        ((1. / V_arcor_sys_np) *
            pow(((dcb_c_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
                     (v_tisscor_ / V_arcor_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arcor_sys_np)) *
                        (dcb_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
                            (v_tisscor_ / V_arcor_sys_np) * dct_o2_dpp_o2(ppO2_arcor_sys_np)) -
                    dcb_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np)),
                -1.) *
            (dcb_c_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) * 1.0 *
                    (cb_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) -
                        cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                dcb_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) * 1.0 *
                    (cb_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) -
                        cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np))));
    // w.r.t. upstream ppCO2
    wkstiff(69, 58) =
        theta *
        (-(1. / V_arcor_sys_np) *
            pow(((dcb_c_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
                     (v_tisscor_ / V_arcor_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arcor_sys_np)) *
                        (dcb_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
                            (v_tisscor_ / V_arcor_sys_np) * dct_o2_dpp_o2(ppO2_arcor_sys_np)) -
                    dcb_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np)),
                -1.) *
            ((dcb_c_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
                 (v_tisscor_ / V_arcor_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arcor_sys_np)) *
                    q_arcor_sys_in_np * dcb_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) -
                dcb_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) * q_arcor_sys_in_np *
                    dcb_c_o2_dpp_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)));
    // w.r.t. upstream ppO2
    wkstiff(69, 59) =
        theta *
        (-(1. / V_arcor_sys_np) *
            pow(((dcb_c_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
                     (v_tisscor_ / V_arcor_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arcor_sys_np)) *
                        (dcb_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
                            (v_tisscor_ / V_arcor_sys_np) * dct_o2_dpp_o2(ppO2_arcor_sys_np)) -
                    dcb_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np)),
                -1.) *
            ((dcb_c_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
                 (v_tisscor_ / V_arcor_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arcor_sys_np)) *
                    q_arcor_sys_in_np * dcb_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np) -
                dcb_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) * q_arcor_sys_in_np *
                    dcb_c_o2_dpp_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)));
    // w.r.t. ppCO2
    wkstiff(69, 68) =
        theta *
        (-(1. / V_arcor_sys_np) *
                pow(((dcb_c_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
                         (v_tisscor_ / V_arcor_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arcor_sys_np)) *
                            (dcb_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
                                (v_tisscor_ / V_arcor_sys_np) * dct_o2_dpp_o2(ppO2_arcor_sys_np)) -
                        dcb_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) *
                            dcb_c_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np)),
                    -2.) *
                ((d2cb_c_o2_dpp_c_o22(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
                     (v_tisscor_ / V_arcor_sys_np) * d2ct_c_o2_dpp_c_o22(ppCO2_arcor_sys_np)) *
                        (dcb_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
                            (v_tisscor_ / V_arcor_sys_np) * dct_o2_dpp_o2(ppO2_arcor_sys_np)) +
                    (dcb_c_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
                        (v_tisscor_ / V_arcor_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arcor_sys_np)) *
                        d2cb_o2_dpp_o2dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) -
                    d2cb_o2_dpp_c_o22(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) -
                    dcb_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) *
                        d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np)) *
                ((dcb_c_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
                     (v_tisscor_ / V_arcor_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arcor_sys_np)) *
                        (q_arcor_sys_in_np * (cb_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) -
                                                 cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) +
                            m_o2_arcor_ * ct_o2(ppO2_arcor_sys_np) /
                                (beta + ct_o2(ppO2_arcor_sys_np))) -
                    dcb_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) *
                        (q_arcor_sys_in_np * (cb_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) -
                                                 cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                            m_c_o2_arcor_)) +
            (1. / V_arcor_sys_np) *
                pow(((dcb_c_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
                         (v_tisscor_ / V_arcor_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arcor_sys_np)) *
                            (dcb_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
                                (v_tisscor_ / V_arcor_sys_np) * dct_o2_dpp_o2(ppO2_arcor_sys_np)) -
                        dcb_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) *
                            dcb_c_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np)),
                    -1.) *
                ((d2cb_c_o2_dpp_c_o22(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
                     (v_tisscor_ / V_arcor_sys_np) * d2ct_c_o2_dpp_c_o22(ppCO2_arcor_sys_np)) *
                        (q_arcor_sys_in_np * (cb_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) -
                                                 cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) +
                            m_o2_arcor_ * ct_o2(ppO2_arcor_sys_np) /
                                (beta + ct_o2(ppO2_arcor_sys_np))) +
                    (dcb_c_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
                        (v_tisscor_ / V_arcor_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arcor_sys_np)) *
                        q_arcor_sys_in_np * dcb_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) -
                    d2cb_o2_dpp_c_o22(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) *
                        (q_arcor_sys_in_np * (cb_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) -
                                                 cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                            m_c_o2_arcor_) -
                    dcb_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) * q_arcor_sys_in_np *
                        dcb_c_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np)));
    // w.r.t. ppO2
    wkstiff(69, 69) =
        1. / ts_size +
        theta *
            (-(1. / V_arcor_sys_np) *
                    pow(((dcb_c_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
                             (v_tisscor_ / V_arcor_sys_np) *
                                 dct_c_o2_dpp_c_o2(ppCO2_arcor_sys_np)) *
                                (dcb_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
                                    (v_tisscor_ / V_arcor_sys_np) *
                                        dct_o2_dpp_o2(ppO2_arcor_sys_np)) -
                            dcb_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np)),
                        -2.) *
                    (d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) *
                            (dcb_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
                                (v_tisscor_ / V_arcor_sys_np) * dct_o2_dpp_o2(ppO2_arcor_sys_np)) +
                        (dcb_c_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
                            (v_tisscor_ / V_arcor_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arcor_sys_np)) *
                            (d2cb_o2_dpp_o22(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
                                (v_tisscor_ / V_arcor_sys_np) *
                                    d2ct_o2_dpp_o22(ppO2_arcor_sys_np)) -
                        d2cb_o2_dpp_o2dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) *
                            dcb_c_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) -
                        dcb_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) *
                            d2cb_c_o2_dpp_o22(ppCO2_arcor_sys_np, ppO2_arcor_sys_np)) *
                    ((dcb_c_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
                         (v_tisscor_ / V_arcor_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arcor_sys_np)) *
                            (q_arcor_sys_in_np * (cb_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) -
                                                     cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) +
                                m_o2_arcor_ * ct_o2(ppO2_arcor_sys_np) /
                                    (beta + ct_o2(ppO2_arcor_sys_np))) -
                        dcb_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) *
                            (q_arcor_sys_in_np * (cb_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) -
                                                     cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                                m_c_o2_arcor_)) +
                (1. / V_arcor_sys_np) *
                    pow(((dcb_c_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
                             (v_tisscor_ / V_arcor_sys_np) *
                                 dct_c_o2_dpp_c_o2(ppCO2_arcor_sys_np)) *
                                (dcb_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
                                    (v_tisscor_ / V_arcor_sys_np) *
                                        dct_o2_dpp_o2(ppO2_arcor_sys_np)) -
                            dcb_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np)),
                        -1.) *
                    (d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) *
                            (q_arcor_sys_in_np * (cb_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) -
                                                     cb_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) +
                                m_o2_arcor_ * ct_o2(ppO2_arcor_sys_np) /
                                    (beta + ct_o2(ppO2_arcor_sys_np))) +
                        (dcb_c_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
                            (v_tisscor_ / V_arcor_sys_np) * dct_c_o2_dpp_c_o2(ppCO2_arcor_sys_np)) *
                            (q_arcor_sys_in_np *
                                    dcb_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) +
                                dct_o2_dpp_o2(ppO2_arcor_sys_np) * beta /
                                    ((beta + ct_o2(ppO2_arcor_sys_np)) *
                                        (beta + ct_o2(ppO2_arcor_sys_np)))) -
                        d2cb_o2_dpp_o2dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) *
                            (q_arcor_sys_in_np * (cb_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) -
                                                     cb_c_o2(ppCO2_ar_sys_np, ppO2_ar_sys_np)) -
                                m_c_o2_arcor_) -
                        dcb_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) * q_arcor_sys_in_np *
                            dcb_c_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np)));



    //////// systemic splanchnic veins CO2
    // w.r.t. upstream flux
    wkstiff(70, 7) =
        theta * ((1. / V_venspl_sys_np) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                                dcb_o2_dpp_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) -
                            dcb_o2_dpp_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np)),
                        -1.) *
                    (dcb_o2_dpp_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                            (1.0 * (cb_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) -
                                       cb_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np))) -
                        dcb_c_o2_dpp_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                            (1.0 * (cb_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) -
                                       cb_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np)))));
    // w.r.t. mech. pressure
    wkstiff(70, 12) =
        theta * (c_venspl_sys_ * (-1. / (V_venspl_sys_np * V_venspl_sys_np)) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                                dcb_o2_dpp_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) -
                            dcb_o2_dpp_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np)),
                        -1.) *
                    (dcb_o2_dpp_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                            (q_arspl_sys_np * (cb_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) -
                                                  cb_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np))) -
                        dcb_c_o2_dpp_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                            (q_arspl_sys_np * (cb_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) -
                                                  cb_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np)))));
    // w.r.t. upstream ppCO2
    wkstiff(70, 60) =
        theta * (-(1. / V_venspl_sys_np) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                                dcb_o2_dpp_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) -
                            dcb_o2_dpp_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np)),
                        -1.) *
                    (dcb_o2_dpp_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) * q_arspl_sys_np *
                            dcb_c_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) -
                        dcb_c_o2_dpp_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) * q_arspl_sys_np *
                            dcb_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np)));
    // w.r.t. upstream ppO2
    wkstiff(70, 61) =
        theta * (-(1. / V_venspl_sys_np) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                                dcb_o2_dpp_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) -
                            dcb_o2_dpp_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np)),
                        -1.) *
                    (dcb_o2_dpp_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) * q_arspl_sys_np *
                            dcb_c_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) -
                        dcb_c_o2_dpp_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) * q_arspl_sys_np *
                            dcb_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np)));
    // w.r.t. ppCO2
    wkstiff(70, 70) =
        1. / ts_size +
        theta *
            (-(1. / V_venspl_sys_np) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                                dcb_o2_dpp_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) -
                            dcb_o2_dpp_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np)),
                        -2.) *
                    (d2cb_c_o2_dpp_c_o22(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                            dcb_o2_dpp_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) +
                        dcb_c_o2_dpp_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                            d2cb_o2_dpp_o2dpp_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) -
                        d2cb_o2_dpp_c_o22(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                            dcb_c_o2_dpp_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) -
                        dcb_o2_dpp_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                            d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np)) *
                    (dcb_o2_dpp_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                            (q_arspl_sys_np * (cb_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) -
                                                  cb_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np))) -
                        dcb_c_o2_dpp_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                            (q_arspl_sys_np * (cb_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) -
                                                  cb_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np)))) +
                (1. / V_venspl_sys_np) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                                dcb_o2_dpp_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) -
                            dcb_o2_dpp_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np)),
                        -1.) *
                    (d2cb_o2_dpp_o2dpp_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                            (q_arspl_sys_np * (cb_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) -
                                                  cb_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np))) +
                        dcb_o2_dpp_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) * q_arspl_sys_np *
                            dcb_c_o2_dpp_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) -
                        d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                            (q_arspl_sys_np * (cb_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) -
                                                  cb_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np))) -
                        dcb_c_o2_dpp_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) * q_arspl_sys_np *
                            dcb_o2_dpp_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np)));
    // w.r.t. ppO2
    wkstiff(70, 71) =
        theta *
        (-(1. / V_venspl_sys_np) *
                pow((dcb_c_o2_dpp_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                            dcb_o2_dpp_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) -
                        dcb_o2_dpp_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                            dcb_c_o2_dpp_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np)),
                    -2.) *
                (d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                        dcb_o2_dpp_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) +
                    dcb_c_o2_dpp_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                        d2cb_o2_dpp_o22(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) -
                    d2cb_o2_dpp_o2dpp_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) -
                    dcb_o2_dpp_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                        d2cb_c_o2_dpp_o22(ppCO2_venspl_sys_np, ppO2_venspl_sys_np)) *
                (dcb_o2_dpp_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                        (q_arspl_sys_np * (cb_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) -
                                              cb_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np))) -
                    dcb_c_o2_dpp_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                        (q_arspl_sys_np * (cb_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) -
                                              cb_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np)))) +
            (1. / V_venspl_sys_np) *
                pow((dcb_c_o2_dpp_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                            dcb_o2_dpp_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) -
                        dcb_o2_dpp_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                            dcb_c_o2_dpp_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np)),
                    -1.) *
                (d2cb_o2_dpp_o22(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                        (q_arspl_sys_np * (cb_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) -
                                              cb_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np))) +
                    dcb_o2_dpp_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) * q_arspl_sys_np *
                        dcb_c_o2_dpp_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) -
                    d2cb_c_o2_dpp_o22(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                        (q_arspl_sys_np * (cb_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) -
                                              cb_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np))) -
                    dcb_c_o2_dpp_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) * q_arspl_sys_np *
                        dcb_o2_dpp_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np)));

    //////// systemic splanchnic veins O2
    // w.r.t. upstream flux
    wkstiff(71, 7) =
        theta * ((1. / V_venspl_sys_np) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                                dcb_o2_dpp_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) -
                            dcb_o2_dpp_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np)),
                        -1.) *
                    (dcb_c_o2_dpp_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                            (1.0 * (cb_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) -
                                       cb_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np))) -
                        dcb_o2_dpp_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                            (1.0 * (cb_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) -
                                       cb_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np)))));
    // w.r.t. mech. pressure
    wkstiff(71, 12) =
        theta *
        (c_venspl_sys_ * (-1. / (V_venspl_sys_np * V_venspl_sys_np)) *
            pow((dcb_c_o2_dpp_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                        dcb_o2_dpp_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) -
                    dcb_o2_dpp_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np)),
                -1.) *
            (dcb_c_o2_dpp_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                    (q_arspl_sys_np * (cb_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) -
                                          cb_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np))) -
                dcb_o2_dpp_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                    (q_arspl_sys_np * (cb_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) -
                                          cb_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np)))));
    // w.r.t. upstream ppCO2
    wkstiff(71, 60) =
        theta * (-(1. / V_venspl_sys_np) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                                dcb_o2_dpp_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) -
                            dcb_o2_dpp_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np)),
                        -1.) *
                    (dcb_c_o2_dpp_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) * q_arspl_sys_np *
                            dcb_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) -
                        dcb_o2_dpp_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) * q_arspl_sys_np *
                            dcb_c_o2_dpp_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np)));
    // w.r.t. upstream ppO2
    wkstiff(71, 61) =
        theta * (-(1. / V_venspl_sys_np) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                                dcb_o2_dpp_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) -
                            dcb_o2_dpp_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np)),
                        -1.) *
                    (dcb_c_o2_dpp_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) * q_arspl_sys_np *
                            dcb_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np) -
                        dcb_o2_dpp_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) * q_arspl_sys_np *
                            dcb_c_o2_dpp_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np)));
    // w.r.t. ppCO2
    wkstiff(71, 70) =
        theta *
        (-(1. / V_venspl_sys_np) *
                pow((dcb_c_o2_dpp_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                            dcb_o2_dpp_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) -
                        dcb_o2_dpp_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                            dcb_c_o2_dpp_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np)),
                    -2.) *
                (d2cb_c_o2_dpp_c_o22(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                        dcb_o2_dpp_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) +
                    dcb_c_o2_dpp_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                        d2cb_o2_dpp_o2dpp_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) -
                    d2cb_o2_dpp_c_o22(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) -
                    dcb_o2_dpp_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                        d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np)) *
                (dcb_c_o2_dpp_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                        (q_arspl_sys_np * (cb_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) -
                                              cb_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np))) -
                    dcb_o2_dpp_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                        (q_arspl_sys_np * (cb_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) -
                                              cb_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np)))) +
            (1. / V_venspl_sys_np) *
                pow((dcb_c_o2_dpp_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                            dcb_o2_dpp_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) -
                        dcb_o2_dpp_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                            dcb_c_o2_dpp_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np)),
                    -1.) *
                (d2cb_c_o2_dpp_c_o22(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                        (q_arspl_sys_np * (cb_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) -
                                              cb_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np))) +
                    dcb_c_o2_dpp_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) * q_arspl_sys_np *
                        dcb_o2_dpp_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) -
                    d2cb_o2_dpp_c_o22(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                        (q_arspl_sys_np * (cb_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) -
                                              cb_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np))) -
                    dcb_o2_dpp_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) * q_arspl_sys_np *
                        dcb_c_o2_dpp_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np)));
    // w.r.t. ppO2
    wkstiff(71, 71) =
        1. / ts_size +
        theta *
            (-(1. / V_venspl_sys_np) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                                dcb_o2_dpp_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) -
                            dcb_o2_dpp_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np)),
                        -2.) *
                    (d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                            dcb_o2_dpp_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) +
                        dcb_c_o2_dpp_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                            d2cb_o2_dpp_o22(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) -
                        d2cb_o2_dpp_o2dpp_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                            dcb_c_o2_dpp_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) -
                        dcb_o2_dpp_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                            d2cb_c_o2_dpp_o22(ppCO2_venspl_sys_np, ppO2_venspl_sys_np)) *
                    (dcb_c_o2_dpp_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                            (q_arspl_sys_np * (cb_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) -
                                                  cb_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np))) -
                        dcb_o2_dpp_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                            (q_arspl_sys_np *
                                (cb_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) -
                                    cb_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np)))) +
                (1. / V_venspl_sys_np) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                                dcb_o2_dpp_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) -
                            dcb_o2_dpp_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np)),
                        -1.) *
                    (d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                            (q_arspl_sys_np * (cb_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) -
                                                  cb_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np))) +
                        dcb_c_o2_dpp_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                            q_arspl_sys_np *
                            dcb_o2_dpp_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) -
                        d2cb_o2_dpp_o2dpp_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) *
                            (q_arspl_sys_np * (cb_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) -
                                                  cb_c_o2(ppCO2_arspl_sys_np, ppO2_arspl_sys_np))) -
                        dcb_o2_dpp_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) * q_arspl_sys_np *
                            dcb_c_o2_dpp_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np)));


    //////// systemic extra-splanchnic veins CO2
    // w.r.t. upstream flux
    wkstiff(72, 8) =
        theta * ((1. / V_venespl_sys_np) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                                dcb_o2_dpp_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) -
                            dcb_o2_dpp_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np)),
                        -1.) *
                    (dcb_o2_dpp_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                            (1.0 * (cb_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) -
                                       cb_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np))) -
                        dcb_c_o2_dpp_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                            (1.0 * (cb_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) -
                                       cb_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np)))));
    // w.r.t. mech. pressure
    wkstiff(72, 14) =
        theta *
        (c_venespl_sys_ * (-1. / (V_venespl_sys_np * V_venespl_sys_np)) *
            pow((dcb_c_o2_dpp_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                        dcb_o2_dpp_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) -
                    dcb_o2_dpp_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np)),
                -1.) *
            (dcb_o2_dpp_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                    (q_arespl_sys_np * (cb_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) -
                                           cb_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np))) -
                dcb_c_o2_dpp_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                    (q_arespl_sys_np * (cb_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) -
                                           cb_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np)))));
    // w.r.t. upstream ppCO2
    wkstiff(72, 62) =
        theta *
        (-(1. / V_venespl_sys_np) *
            pow((dcb_c_o2_dpp_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                        dcb_o2_dpp_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) -
                    dcb_o2_dpp_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np)),
                -1.) *
            (dcb_o2_dpp_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) * q_arespl_sys_np *
                    dcb_c_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) -
                dcb_c_o2_dpp_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) * q_arespl_sys_np *
                    dcb_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np)));
    // w.r.t. upstream ppO2
    wkstiff(72, 63) =
        theta *
        (-(1. / V_venespl_sys_np) *
            pow((dcb_c_o2_dpp_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                        dcb_o2_dpp_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) -
                    dcb_o2_dpp_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np)),
                -1.) *
            (dcb_o2_dpp_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) * q_arespl_sys_np *
                    dcb_c_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) -
                dcb_c_o2_dpp_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) * q_arespl_sys_np *
                    dcb_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np)));
    // w.r.t. ppCO2
    wkstiff(72, 72) =
        1. / ts_size +
        theta *
            (-(1. / V_venespl_sys_np) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                                dcb_o2_dpp_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) -
                            dcb_o2_dpp_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np)),
                        -2.) *
                    (d2cb_c_o2_dpp_c_o22(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                            dcb_o2_dpp_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) +
                        dcb_c_o2_dpp_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                            d2cb_o2_dpp_o2dpp_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) -
                        d2cb_o2_dpp_c_o22(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                            dcb_c_o2_dpp_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) -
                        dcb_o2_dpp_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                            d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np)) *
                    (dcb_o2_dpp_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                            (q_arespl_sys_np *
                                (cb_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) -
                                    cb_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np))) -
                        dcb_c_o2_dpp_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                            (q_arespl_sys_np *
                                (cb_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) -
                                    cb_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np)))) +
                (1. / V_venespl_sys_np) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                                dcb_o2_dpp_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) -
                            dcb_o2_dpp_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np)),
                        -1.) *
                    (d2cb_o2_dpp_o2dpp_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                            (q_arespl_sys_np *
                                (cb_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) -
                                    cb_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np))) +
                        dcb_o2_dpp_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) * q_arespl_sys_np *
                            dcb_c_o2_dpp_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) -
                        d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                            (q_arespl_sys_np *
                                (cb_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) -
                                    cb_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np))) -
                        dcb_c_o2_dpp_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                            q_arespl_sys_np *
                            dcb_o2_dpp_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np)));
    // w.r.t. ppO2
    wkstiff(72, 73) =
        theta *
        (-(1. / V_venespl_sys_np) *
                pow((dcb_c_o2_dpp_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                            dcb_o2_dpp_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) -
                        dcb_o2_dpp_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                            dcb_c_o2_dpp_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np)),
                    -2.) *
                (d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                        dcb_o2_dpp_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) +
                    dcb_c_o2_dpp_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                        d2cb_o2_dpp_o22(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) -
                    d2cb_o2_dpp_o2dpp_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) -
                    dcb_o2_dpp_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                        d2cb_c_o2_dpp_o22(ppCO2_venespl_sys_np, ppO2_venespl_sys_np)) *
                (dcb_o2_dpp_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                        (q_arespl_sys_np * (cb_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) -
                                               cb_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np))) -
                    dcb_c_o2_dpp_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                        (q_arespl_sys_np * (cb_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) -
                                               cb_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np)))) +
            (1. / V_venespl_sys_np) *
                pow((dcb_c_o2_dpp_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                            dcb_o2_dpp_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) -
                        dcb_o2_dpp_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                            dcb_c_o2_dpp_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np)),
                    -1.) *
                (d2cb_o2_dpp_o22(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                        (q_arespl_sys_np * (cb_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) -
                                               cb_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np))) +
                    dcb_o2_dpp_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) * q_arespl_sys_np *
                        dcb_c_o2_dpp_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) -
                    d2cb_c_o2_dpp_o22(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                        (q_arespl_sys_np * (cb_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) -
                                               cb_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np))) -
                    dcb_c_o2_dpp_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) * q_arespl_sys_np *
                        dcb_o2_dpp_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np)));

    //////// systemic extra-splanchnic veins O2
    // w.r.t. upstream flux
    wkstiff(73, 8) =
        theta * ((1. / V_venespl_sys_np) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                                dcb_o2_dpp_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) -
                            dcb_o2_dpp_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np)),
                        -1.) *
                    (dcb_c_o2_dpp_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                            (1.0 * (cb_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) -
                                       cb_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np))) -
                        dcb_o2_dpp_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                            (1.0 * (cb_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) -
                                       cb_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np)))));
    // w.r.t. mech. pressure
    wkstiff(73, 14) =
        theta *
        (c_venespl_sys_ * (-1. / (V_venespl_sys_np * V_venespl_sys_np)) *
            pow((dcb_c_o2_dpp_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                        dcb_o2_dpp_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) -
                    dcb_o2_dpp_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np)),
                -1.) *
            (dcb_c_o2_dpp_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                    (q_arespl_sys_np * (cb_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) -
                                           cb_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np))) -
                dcb_o2_dpp_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                    (q_arespl_sys_np * (cb_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) -
                                           cb_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np)))));
    // w.r.t. upstream ppCO2
    wkstiff(73, 62) =
        theta *
        (-(1. / V_venespl_sys_np) *
            pow((dcb_c_o2_dpp_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                        dcb_o2_dpp_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) -
                    dcb_o2_dpp_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np)),
                -1.) *
            (dcb_c_o2_dpp_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) * q_arespl_sys_np *
                    dcb_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) -
                dcb_o2_dpp_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) * q_arespl_sys_np *
                    dcb_c_o2_dpp_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np)));
    // w.r.t. upstream ppO2
    wkstiff(73, 63) =
        theta *
        (-(1. / V_venespl_sys_np) *
            pow((dcb_c_o2_dpp_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                        dcb_o2_dpp_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) -
                    dcb_o2_dpp_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np)),
                -1.) *
            (dcb_c_o2_dpp_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) * q_arespl_sys_np *
                    dcb_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np) -
                dcb_o2_dpp_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) * q_arespl_sys_np *
                    dcb_c_o2_dpp_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np)));
    // w.r.t. ppCO2
    wkstiff(73, 72) =
        theta *
        (-(1. / V_venespl_sys_np) *
                pow((dcb_c_o2_dpp_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                            dcb_o2_dpp_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) -
                        dcb_o2_dpp_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                            dcb_c_o2_dpp_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np)),
                    -2.) *
                (d2cb_c_o2_dpp_c_o22(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                        dcb_o2_dpp_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) +
                    dcb_c_o2_dpp_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                        d2cb_o2_dpp_o2dpp_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) -
                    d2cb_o2_dpp_c_o22(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) -
                    dcb_o2_dpp_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                        d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np)) *
                (dcb_c_o2_dpp_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                        (q_arespl_sys_np * (cb_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) -
                                               cb_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np))) -
                    dcb_o2_dpp_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                        (q_arespl_sys_np * (cb_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) -
                                               cb_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np)))) +
            (1. / V_venespl_sys_np) *
                pow((dcb_c_o2_dpp_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                            dcb_o2_dpp_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) -
                        dcb_o2_dpp_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                            dcb_c_o2_dpp_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np)),
                    -1.) *
                (d2cb_c_o2_dpp_c_o22(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                        (q_arespl_sys_np * (cb_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) -
                                               cb_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np))) +
                    dcb_c_o2_dpp_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) * q_arespl_sys_np *
                        dcb_o2_dpp_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) -
                    d2cb_o2_dpp_c_o22(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                        (q_arespl_sys_np * (cb_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) -
                                               cb_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np))) -
                    dcb_o2_dpp_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) * q_arespl_sys_np *
                        dcb_c_o2_dpp_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np)));
    // w.r.t. ppO2
    wkstiff(73, 73) =
        1. / ts_size +
        theta * (-(1. / V_venespl_sys_np) *
                        pow((dcb_c_o2_dpp_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                                    dcb_o2_dpp_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) -
                                dcb_o2_dpp_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                                    dcb_c_o2_dpp_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np)),
                            -2.) *
                        (d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                                dcb_o2_dpp_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) +
                            dcb_c_o2_dpp_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                                d2cb_o2_dpp_o22(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) -
                            d2cb_o2_dpp_o2dpp_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) -
                            dcb_o2_dpp_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                                d2cb_c_o2_dpp_o22(ppCO2_venespl_sys_np, ppO2_venespl_sys_np)) *
                        (dcb_c_o2_dpp_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                                (q_arespl_sys_np *
                                    (cb_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) -
                                        cb_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np))) -
                            dcb_o2_dpp_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                                (q_arespl_sys_np *
                                    (cb_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) -
                                        cb_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np)))) +
                    (1. / V_venespl_sys_np) *
                        pow((dcb_c_o2_dpp_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                                    dcb_o2_dpp_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) -
                                dcb_o2_dpp_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                                    dcb_c_o2_dpp_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np)),
                            -1.) *
                        (d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                                (q_arespl_sys_np *
                                    (cb_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) -
                                        cb_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np))) +
                            dcb_c_o2_dpp_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                                q_arespl_sys_np *
                                dcb_o2_dpp_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) -
                            d2cb_o2_dpp_o2dpp_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                                (q_arespl_sys_np *
                                    (cb_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) -
                                        cb_c_o2(ppCO2_arespl_sys_np, ppO2_arespl_sys_np))) -
                            dcb_o2_dpp_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) *
                                q_arespl_sys_np *
                                dcb_c_o2_dpp_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np)));


    //////// systemic muscular veins CO2
    // w.r.t. upstream flux
    wkstiff(74, 9) =
        theta * ((1. / V_venmsc_sys_np) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                                dcb_o2_dpp_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) -
                            dcb_o2_dpp_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np)),
                        -1.) *
                    (dcb_o2_dpp_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                            (1.0 * (cb_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) -
                                       cb_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np))) -
                        dcb_c_o2_dpp_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                            (1.0 * (cb_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) -
                                       cb_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np)))));
    // w.r.t. mech. pressure
    wkstiff(74, 16) =
        theta * (c_venmsc_sys_ * (-1. / (V_venmsc_sys_np * V_venmsc_sys_np)) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                                dcb_o2_dpp_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) -
                            dcb_o2_dpp_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np)),
                        -1.) *
                    (dcb_o2_dpp_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                            (q_armsc_sys_np * (cb_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) -
                                                  cb_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np))) -
                        dcb_c_o2_dpp_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                            (q_armsc_sys_np * (cb_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) -
                                                  cb_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np)))));
    // w.r.t. upstream ppCO2
    wkstiff(74, 64) =
        theta * (-(1. / V_venmsc_sys_np) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                                dcb_o2_dpp_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) -
                            dcb_o2_dpp_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np)),
                        -1.) *
                    (dcb_o2_dpp_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) * q_armsc_sys_np *
                            dcb_c_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) -
                        dcb_c_o2_dpp_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) * q_armsc_sys_np *
                            dcb_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np)));
    // w.r.t. upstream ppO2
    wkstiff(74, 65) =
        theta * (-(1. / V_venmsc_sys_np) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                                dcb_o2_dpp_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) -
                            dcb_o2_dpp_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np)),
                        -1.) *
                    (dcb_o2_dpp_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) * q_armsc_sys_np *
                            dcb_c_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) -
                        dcb_c_o2_dpp_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) * q_armsc_sys_np *
                            dcb_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np)));
    // w.r.t. ppCO2
    wkstiff(74, 74) =
        1. / ts_size +
        theta *
            (-(1. / V_venmsc_sys_np) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                                dcb_o2_dpp_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) -
                            dcb_o2_dpp_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np)),
                        -2.) *
                    (d2cb_c_o2_dpp_c_o22(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                            dcb_o2_dpp_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) +
                        dcb_c_o2_dpp_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                            d2cb_o2_dpp_o2dpp_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) -
                        d2cb_o2_dpp_c_o22(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                            dcb_c_o2_dpp_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) -
                        dcb_o2_dpp_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                            d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np)) *
                    (dcb_o2_dpp_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                            (q_armsc_sys_np * (cb_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) -
                                                  cb_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np))) -
                        dcb_c_o2_dpp_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                            (q_armsc_sys_np * (cb_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) -
                                                  cb_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np)))) +
                (1. / V_venmsc_sys_np) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                                dcb_o2_dpp_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) -
                            dcb_o2_dpp_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np)),
                        -1.) *
                    (d2cb_o2_dpp_o2dpp_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                            (q_armsc_sys_np * (cb_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) -
                                                  cb_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np))) +
                        dcb_o2_dpp_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) * q_armsc_sys_np *
                            dcb_c_o2_dpp_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) -
                        d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                            (q_armsc_sys_np * (cb_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) -
                                                  cb_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np))) -
                        dcb_c_o2_dpp_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) * q_armsc_sys_np *
                            dcb_o2_dpp_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np)));
    // w.r.t. ppO2
    wkstiff(74, 75) =
        theta *
        (-(1. / V_venmsc_sys_np) *
                pow((dcb_c_o2_dpp_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                            dcb_o2_dpp_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) -
                        dcb_o2_dpp_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                            dcb_c_o2_dpp_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np)),
                    -2.) *
                (d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                        dcb_o2_dpp_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) +
                    dcb_c_o2_dpp_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                        d2cb_o2_dpp_o22(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) -
                    d2cb_o2_dpp_o2dpp_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) -
                    dcb_o2_dpp_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                        d2cb_c_o2_dpp_o22(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np)) *
                (dcb_o2_dpp_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                        (q_armsc_sys_np * (cb_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) -
                                              cb_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np))) -
                    dcb_c_o2_dpp_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                        (q_armsc_sys_np * (cb_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) -
                                              cb_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np)))) +
            (1. / V_venmsc_sys_np) *
                pow((dcb_c_o2_dpp_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                            dcb_o2_dpp_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) -
                        dcb_o2_dpp_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                            dcb_c_o2_dpp_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np)),
                    -1.) *
                (d2cb_o2_dpp_o22(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                        (q_armsc_sys_np * (cb_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) -
                                              cb_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np))) +
                    dcb_o2_dpp_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) * q_armsc_sys_np *
                        dcb_c_o2_dpp_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) -
                    d2cb_c_o2_dpp_o22(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                        (q_armsc_sys_np * (cb_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) -
                                              cb_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np))) -
                    dcb_c_o2_dpp_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) * q_armsc_sys_np *
                        dcb_o2_dpp_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np)));

    //////// systemic muscular veins O2
    // w.r.t. upstream flux
    wkstiff(75, 9) =
        theta * ((1. / V_venmsc_sys_np) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                                dcb_o2_dpp_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) -
                            dcb_o2_dpp_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np)),
                        -1.) *
                    (dcb_c_o2_dpp_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                            (1.0 * (cb_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) -
                                       cb_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np))) -
                        dcb_o2_dpp_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                            (1.0 * (cb_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) -
                                       cb_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np)))));
    // w.r.t. mech. pressure
    wkstiff(75, 16) =
        theta *
        (c_venmsc_sys_ * (-1. / (V_venmsc_sys_np * V_venmsc_sys_np)) *
            pow((dcb_c_o2_dpp_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                        dcb_o2_dpp_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) -
                    dcb_o2_dpp_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np)),
                -1.) *
            (dcb_c_o2_dpp_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                    (q_armsc_sys_np * (cb_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) -
                                          cb_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np))) -
                dcb_o2_dpp_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                    (q_armsc_sys_np * (cb_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) -
                                          cb_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np)))));
    // w.r.t. upstream ppCO2
    wkstiff(75, 64) =
        theta * (-(1. / V_venmsc_sys_np) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                                dcb_o2_dpp_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) -
                            dcb_o2_dpp_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np)),
                        -1.) *
                    (dcb_c_o2_dpp_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) * q_armsc_sys_np *
                            dcb_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) -
                        dcb_o2_dpp_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) * q_armsc_sys_np *
                            dcb_c_o2_dpp_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np)));
    // w.r.t. upstream ppO2
    wkstiff(75, 65) =
        theta * (-(1. / V_venmsc_sys_np) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                                dcb_o2_dpp_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) -
                            dcb_o2_dpp_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np)),
                        -1.) *
                    (dcb_c_o2_dpp_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) * q_armsc_sys_np *
                            dcb_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np) -
                        dcb_o2_dpp_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) * q_armsc_sys_np *
                            dcb_c_o2_dpp_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np)));
    // w.r.t. ppCO2
    wkstiff(75, 74) =
        theta *
        (-(1. / V_venmsc_sys_np) *
                pow((dcb_c_o2_dpp_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                            dcb_o2_dpp_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) -
                        dcb_o2_dpp_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                            dcb_c_o2_dpp_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np)),
                    -2.) *
                (d2cb_c_o2_dpp_c_o22(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                        dcb_o2_dpp_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) +
                    dcb_c_o2_dpp_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                        d2cb_o2_dpp_o2dpp_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) -
                    d2cb_o2_dpp_c_o22(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) -
                    dcb_o2_dpp_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                        d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np)) *
                (dcb_c_o2_dpp_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                        (q_armsc_sys_np * (cb_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) -
                                              cb_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np))) -
                    dcb_o2_dpp_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                        (q_armsc_sys_np * (cb_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) -
                                              cb_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np)))) +
            (1. / V_venmsc_sys_np) *
                pow((dcb_c_o2_dpp_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                            dcb_o2_dpp_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) -
                        dcb_o2_dpp_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                            dcb_c_o2_dpp_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np)),
                    -1.) *
                (d2cb_c_o2_dpp_c_o22(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                        (q_armsc_sys_np * (cb_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) -
                                              cb_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np))) +
                    dcb_c_o2_dpp_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) * q_armsc_sys_np *
                        dcb_o2_dpp_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) -
                    d2cb_o2_dpp_c_o22(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                        (q_armsc_sys_np * (cb_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) -
                                              cb_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np))) -
                    dcb_o2_dpp_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) * q_armsc_sys_np *
                        dcb_c_o2_dpp_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np)));
    // w.r.t. ppO2
    wkstiff(75, 75) =
        1. / ts_size +
        theta *
            (-(1. / V_venmsc_sys_np) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                                dcb_o2_dpp_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) -
                            dcb_o2_dpp_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np)),
                        -2.) *
                    (d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                            dcb_o2_dpp_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) +
                        dcb_c_o2_dpp_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                            d2cb_o2_dpp_o22(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) -
                        d2cb_o2_dpp_o2dpp_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                            dcb_c_o2_dpp_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) -
                        dcb_o2_dpp_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                            d2cb_c_o2_dpp_o22(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np)) *
                    (dcb_c_o2_dpp_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                            (q_armsc_sys_np * (cb_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) -
                                                  cb_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np))) -
                        dcb_o2_dpp_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                            (q_armsc_sys_np *
                                (cb_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) -
                                    cb_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np)))) +
                (1. / V_venmsc_sys_np) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                                dcb_o2_dpp_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) -
                            dcb_o2_dpp_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np)),
                        -1.) *
                    (d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                            (q_armsc_sys_np * (cb_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) -
                                                  cb_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np))) +
                        dcb_c_o2_dpp_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                            q_armsc_sys_np *
                            dcb_o2_dpp_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) -
                        d2cb_o2_dpp_o2dpp_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) *
                            (q_armsc_sys_np * (cb_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) -
                                                  cb_c_o2(ppCO2_armsc_sys_np, ppO2_armsc_sys_np))) -
                        dcb_o2_dpp_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) * q_armsc_sys_np *
                            dcb_c_o2_dpp_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np)));


    //////// systemic cerebral veins CO2
    // w.r.t. upstream flux
    wkstiff(76, 10) =
        theta * ((1. / V_vencer_sys_np) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                                dcb_o2_dpp_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) -
                            dcb_o2_dpp_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np)),
                        -1.) *
                    (dcb_o2_dpp_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                            (1.0 * (cb_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) -
                                       cb_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np))) -
                        dcb_c_o2_dpp_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                            (1.0 * (cb_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) -
                                       cb_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np)))));
    // w.r.t. mech. pressure
    wkstiff(76, 18) =
        theta * (c_vencer_sys_ * (-1. / (V_vencer_sys_np * V_vencer_sys_np)) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                                dcb_o2_dpp_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) -
                            dcb_o2_dpp_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np)),
                        -1.) *
                    (dcb_o2_dpp_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                            (q_arcer_sys_np * (cb_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) -
                                                  cb_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np))) -
                        dcb_c_o2_dpp_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                            (q_arcer_sys_np * (cb_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) -
                                                  cb_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np)))));
    // w.r.t. upstream ppCO2
    wkstiff(76, 66) =
        theta * (-(1. / V_vencer_sys_np) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                                dcb_o2_dpp_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) -
                            dcb_o2_dpp_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np)),
                        -1.) *
                    (dcb_o2_dpp_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) * q_arcer_sys_np *
                            dcb_c_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) -
                        dcb_c_o2_dpp_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) * q_arcer_sys_np *
                            dcb_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np)));
    // w.r.t. upstream ppO2
    wkstiff(76, 67) =
        theta * (-(1. / V_vencer_sys_np) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                                dcb_o2_dpp_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) -
                            dcb_o2_dpp_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np)),
                        -1.) *
                    (dcb_o2_dpp_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) * q_arcer_sys_np *
                            dcb_c_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) -
                        dcb_c_o2_dpp_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) * q_arcer_sys_np *
                            dcb_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np)));
    // w.r.t. ppCO2
    wkstiff(76, 76) =
        1. / ts_size +
        theta *
            (-(1. / V_vencer_sys_np) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                                dcb_o2_dpp_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) -
                            dcb_o2_dpp_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np)),
                        -2.) *
                    (d2cb_c_o2_dpp_c_o22(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                            dcb_o2_dpp_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) +
                        dcb_c_o2_dpp_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                            d2cb_o2_dpp_o2dpp_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) -
                        d2cb_o2_dpp_c_o22(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                            dcb_c_o2_dpp_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) -
                        dcb_o2_dpp_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                            d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np)) *
                    (dcb_o2_dpp_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                            (q_arcer_sys_np * (cb_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) -
                                                  cb_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np))) -
                        dcb_c_o2_dpp_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                            (q_arcer_sys_np * (cb_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) -
                                                  cb_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np)))) +
                (1. / V_vencer_sys_np) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                                dcb_o2_dpp_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) -
                            dcb_o2_dpp_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np)),
                        -1.) *
                    (d2cb_o2_dpp_o2dpp_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                            (q_arcer_sys_np * (cb_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) -
                                                  cb_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np))) +
                        dcb_o2_dpp_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) * q_arcer_sys_np *
                            dcb_c_o2_dpp_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) -
                        d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                            (q_arcer_sys_np * (cb_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) -
                                                  cb_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np))) -
                        dcb_c_o2_dpp_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) * q_arcer_sys_np *
                            dcb_o2_dpp_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np)));
    // w.r.t. ppO2
    wkstiff(76, 77) =
        theta *
        (-(1. / V_vencer_sys_np) *
                pow((dcb_c_o2_dpp_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                            dcb_o2_dpp_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) -
                        dcb_o2_dpp_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                            dcb_c_o2_dpp_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np)),
                    -2.) *
                (d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                        dcb_o2_dpp_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) +
                    dcb_c_o2_dpp_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                        d2cb_o2_dpp_o22(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) -
                    d2cb_o2_dpp_o2dpp_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) -
                    dcb_o2_dpp_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                        d2cb_c_o2_dpp_o22(ppCO2_vencer_sys_np, ppO2_vencer_sys_np)) *
                (dcb_o2_dpp_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                        (q_arcer_sys_np * (cb_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) -
                                              cb_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np))) -
                    dcb_c_o2_dpp_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                        (q_arcer_sys_np * (cb_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) -
                                              cb_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np)))) +
            (1. / V_vencer_sys_np) *
                pow((dcb_c_o2_dpp_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                            dcb_o2_dpp_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) -
                        dcb_o2_dpp_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                            dcb_c_o2_dpp_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np)),
                    -1.) *
                (d2cb_o2_dpp_o22(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                        (q_arcer_sys_np * (cb_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) -
                                              cb_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np))) +
                    dcb_o2_dpp_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) * q_arcer_sys_np *
                        dcb_c_o2_dpp_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) -
                    d2cb_c_o2_dpp_o22(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                        (q_arcer_sys_np * (cb_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) -
                                              cb_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np))) -
                    dcb_c_o2_dpp_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) * q_arcer_sys_np *
                        dcb_o2_dpp_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np)));

    //////// systemic cerebral veins O2
    // w.r.t. upstream flux
    wkstiff(77, 10) =
        theta * ((1. / V_vencer_sys_np) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                                dcb_o2_dpp_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) -
                            dcb_o2_dpp_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np)),
                        -1.) *
                    (dcb_c_o2_dpp_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                            (1.0 * (cb_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) -
                                       cb_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np))) -
                        dcb_o2_dpp_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                            (1.0 * (cb_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) -
                                       cb_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np)))));
    // w.r.t. mech. pressure
    wkstiff(77, 18) =
        theta *
        (c_vencer_sys_ * (-1. / (V_vencer_sys_np * V_vencer_sys_np)) *
            pow((dcb_c_o2_dpp_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                        dcb_o2_dpp_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) -
                    dcb_o2_dpp_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np)),
                -1.) *
            (dcb_c_o2_dpp_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                    (q_arcer_sys_np * (cb_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) -
                                          cb_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np))) -
                dcb_o2_dpp_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                    (q_arcer_sys_np * (cb_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) -
                                          cb_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np)))));
    // w.r.t. upstream ppCO2
    wkstiff(77, 66) =
        theta * (-(1. / V_vencer_sys_np) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                                dcb_o2_dpp_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) -
                            dcb_o2_dpp_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np)),
                        -1.) *
                    (dcb_c_o2_dpp_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) * q_arcer_sys_np *
                            dcb_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) -
                        dcb_o2_dpp_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) * q_arcer_sys_np *
                            dcb_c_o2_dpp_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np)));
    // w.r.t. upstream ppO2
    wkstiff(77, 67) =
        theta * (-(1. / V_vencer_sys_np) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                                dcb_o2_dpp_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) -
                            dcb_o2_dpp_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np)),
                        -1.) *
                    (dcb_c_o2_dpp_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) * q_arcer_sys_np *
                            dcb_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np) -
                        dcb_o2_dpp_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) * q_arcer_sys_np *
                            dcb_c_o2_dpp_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np)));
    // w.r.t. ppCO2
    wkstiff(77, 76) =
        theta *
        (-(1. / V_vencer_sys_np) *
                pow((dcb_c_o2_dpp_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                            dcb_o2_dpp_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) -
                        dcb_o2_dpp_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                            dcb_c_o2_dpp_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np)),
                    -2.) *
                (d2cb_c_o2_dpp_c_o22(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                        dcb_o2_dpp_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) +
                    dcb_c_o2_dpp_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                        d2cb_o2_dpp_o2dpp_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) -
                    d2cb_o2_dpp_c_o22(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) -
                    dcb_o2_dpp_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                        d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np)) *
                (dcb_c_o2_dpp_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                        (q_arcer_sys_np * (cb_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) -
                                              cb_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np))) -
                    dcb_o2_dpp_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                        (q_arcer_sys_np * (cb_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) -
                                              cb_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np)))) +
            (1. / V_vencer_sys_np) *
                pow((dcb_c_o2_dpp_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                            dcb_o2_dpp_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) -
                        dcb_o2_dpp_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                            dcb_c_o2_dpp_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np)),
                    -1.) *
                (d2cb_c_o2_dpp_c_o22(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                        (q_arcer_sys_np * (cb_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) -
                                              cb_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np))) +
                    dcb_c_o2_dpp_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) * q_arcer_sys_np *
                        dcb_o2_dpp_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) -
                    d2cb_o2_dpp_c_o22(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                        (q_arcer_sys_np * (cb_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) -
                                              cb_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np))) -
                    dcb_o2_dpp_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) * q_arcer_sys_np *
                        dcb_c_o2_dpp_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np)));
    // w.r.t. ppO2
    wkstiff(77, 77) =
        1. / ts_size +
        theta *
            (-(1. / V_vencer_sys_np) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                                dcb_o2_dpp_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) -
                            dcb_o2_dpp_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np)),
                        -2.) *
                    (d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                            dcb_o2_dpp_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) +
                        dcb_c_o2_dpp_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                            d2cb_o2_dpp_o22(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) -
                        d2cb_o2_dpp_o2dpp_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                            dcb_c_o2_dpp_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) -
                        dcb_o2_dpp_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                            d2cb_c_o2_dpp_o22(ppCO2_vencer_sys_np, ppO2_vencer_sys_np)) *
                    (dcb_c_o2_dpp_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                            (q_arcer_sys_np * (cb_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) -
                                                  cb_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np))) -
                        dcb_o2_dpp_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                            (q_arcer_sys_np *
                                (cb_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) -
                                    cb_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np)))) +
                (1. / V_vencer_sys_np) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                                dcb_o2_dpp_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) -
                            dcb_o2_dpp_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np)),
                        -1.) *
                    (d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                            (q_arcer_sys_np * (cb_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) -
                                                  cb_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np))) +
                        dcb_c_o2_dpp_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                            q_arcer_sys_np *
                            dcb_o2_dpp_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) -
                        d2cb_o2_dpp_o2dpp_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) *
                            (q_arcer_sys_np * (cb_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) -
                                                  cb_c_o2(ppCO2_arcer_sys_np, ppO2_arcer_sys_np))) -
                        dcb_o2_dpp_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) * q_arcer_sys_np *
                            dcb_c_o2_dpp_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np)));


    //////// systemic coronary veins CO2
    // w.r.t. upstream flux
    wkstiff(78, 11) =
        theta * ((1. / V_vencor_sys_np) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                                dcb_o2_dpp_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) -
                            dcb_o2_dpp_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np)),
                        -1.) *
                    (dcb_o2_dpp_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                            (1.0 * (cb_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) -
                                       cb_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np))) -
                        dcb_c_o2_dpp_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                            (1.0 * (cb_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) -
                                       cb_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np)))));
    // w.r.t. mech. pressure
    wkstiff(78, 20) =
        theta * (c_vencor_sys_ * (-1. / (V_vencor_sys_np * V_vencor_sys_np)) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                                dcb_o2_dpp_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) -
                            dcb_o2_dpp_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np)),
                        -1.) *
                    (dcb_o2_dpp_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                            (q_arcor_sys_np * (cb_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) -
                                                  cb_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np))) -
                        dcb_c_o2_dpp_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                            (q_arcor_sys_np * (cb_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) -
                                                  cb_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np)))));
    // w.r.t. upstream ppCO2
    wkstiff(78, 68) =
        theta * (-(1. / V_vencor_sys_np) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                                dcb_o2_dpp_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) -
                            dcb_o2_dpp_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np)),
                        -1.) *
                    (dcb_o2_dpp_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) * q_arcor_sys_np *
                            dcb_c_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) -
                        dcb_c_o2_dpp_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) * q_arcor_sys_np *
                            dcb_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np)));
    // w.r.t. upstream ppO2
    wkstiff(78, 69) =
        theta * (-(1. / V_vencor_sys_np) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                                dcb_o2_dpp_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) -
                            dcb_o2_dpp_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np)),
                        -1.) *
                    (dcb_o2_dpp_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) * q_arcor_sys_np *
                            dcb_c_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) -
                        dcb_c_o2_dpp_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) * q_arcor_sys_np *
                            dcb_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np)));
    // w.r.t. ppCO2
    wkstiff(78, 78) =
        1. / ts_size +
        theta *
            (-(1. / V_vencor_sys_np) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                                dcb_o2_dpp_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) -
                            dcb_o2_dpp_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np)),
                        -2.) *
                    (d2cb_c_o2_dpp_c_o22(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                            dcb_o2_dpp_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) +
                        dcb_c_o2_dpp_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                            d2cb_o2_dpp_o2dpp_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) -
                        d2cb_o2_dpp_c_o22(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                            dcb_c_o2_dpp_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) -
                        dcb_o2_dpp_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                            d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np)) *
                    (dcb_o2_dpp_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                            (q_arcor_sys_np * (cb_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) -
                                                  cb_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np))) -
                        dcb_c_o2_dpp_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                            (q_arcor_sys_np * (cb_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) -
                                                  cb_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np)))) +
                (1. / V_vencor_sys_np) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                                dcb_o2_dpp_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) -
                            dcb_o2_dpp_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np)),
                        -1.) *
                    (d2cb_o2_dpp_o2dpp_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                            (q_arcor_sys_np * (cb_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) -
                                                  cb_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np))) +
                        dcb_o2_dpp_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) * q_arcor_sys_np *
                            dcb_c_o2_dpp_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) -
                        d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                            (q_arcor_sys_np * (cb_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) -
                                                  cb_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np))) -
                        dcb_c_o2_dpp_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) * q_arcor_sys_np *
                            dcb_o2_dpp_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np)));
    // w.r.t. ppO2
    wkstiff(78, 79) =
        theta *
        (-(1. / V_vencor_sys_np) *
                pow((dcb_c_o2_dpp_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                            dcb_o2_dpp_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) -
                        dcb_o2_dpp_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                            dcb_c_o2_dpp_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np)),
                    -2.) *
                (d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                        dcb_o2_dpp_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) +
                    dcb_c_o2_dpp_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                        d2cb_o2_dpp_o22(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) -
                    d2cb_o2_dpp_o2dpp_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) -
                    dcb_o2_dpp_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                        d2cb_c_o2_dpp_o22(ppCO2_vencor_sys_np, ppO2_vencor_sys_np)) *
                (dcb_o2_dpp_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                        (q_arcor_sys_np * (cb_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) -
                                              cb_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np))) -
                    dcb_c_o2_dpp_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                        (q_arcor_sys_np * (cb_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) -
                                              cb_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np)))) +
            (1. / V_vencor_sys_np) *
                pow((dcb_c_o2_dpp_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                            dcb_o2_dpp_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) -
                        dcb_o2_dpp_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                            dcb_c_o2_dpp_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np)),
                    -1.) *
                (d2cb_o2_dpp_o22(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                        (q_arcor_sys_np * (cb_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) -
                                              cb_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np))) +
                    dcb_o2_dpp_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) * q_arcor_sys_np *
                        dcb_c_o2_dpp_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) -
                    d2cb_c_o2_dpp_o22(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                        (q_arcor_sys_np * (cb_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) -
                                              cb_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np))) -
                    dcb_c_o2_dpp_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) * q_arcor_sys_np *
                        dcb_o2_dpp_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np)));

    //////// systemic coronary veins O2
    // w.r.t. upstream flux
    wkstiff(79, 11) =
        theta * ((1. / V_vencor_sys_np) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                                dcb_o2_dpp_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) -
                            dcb_o2_dpp_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np)),
                        -1.) *
                    (dcb_c_o2_dpp_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                            (1.0 * (cb_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) -
                                       cb_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np))) -
                        dcb_o2_dpp_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                            (1.0 * (cb_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) -
                                       cb_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np)))));
    // w.r.t. mech. pressure
    wkstiff(79, 20) =
        theta *
        (c_vencor_sys_ * (-1. / (V_vencor_sys_np * V_vencor_sys_np)) *
            pow((dcb_c_o2_dpp_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                        dcb_o2_dpp_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) -
                    dcb_o2_dpp_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np)),
                -1.) *
            (dcb_c_o2_dpp_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                    (q_arcor_sys_np * (cb_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) -
                                          cb_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np))) -
                dcb_o2_dpp_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                    (q_arcor_sys_np * (cb_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) -
                                          cb_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np)))));
    // w.r.t. upstream ppCO2
    wkstiff(79, 68) =
        theta * (-(1. / V_vencor_sys_np) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                                dcb_o2_dpp_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) -
                            dcb_o2_dpp_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np)),
                        -1.) *
                    (dcb_c_o2_dpp_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) * q_arcor_sys_np *
                            dcb_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) -
                        dcb_o2_dpp_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) * q_arcor_sys_np *
                            dcb_c_o2_dpp_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np)));
    // w.r.t. upstream ppO2
    wkstiff(79, 69) =
        theta * (-(1. / V_vencor_sys_np) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                                dcb_o2_dpp_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) -
                            dcb_o2_dpp_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np)),
                        -1.) *
                    (dcb_c_o2_dpp_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) * q_arcor_sys_np *
                            dcb_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np) -
                        dcb_o2_dpp_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) * q_arcor_sys_np *
                            dcb_c_o2_dpp_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np)));
    // w.r.t. ppCO2
    wkstiff(79, 78) =
        theta *
        (-(1. / V_vencor_sys_np) *
                pow((dcb_c_o2_dpp_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                            dcb_o2_dpp_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) -
                        dcb_o2_dpp_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                            dcb_c_o2_dpp_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np)),
                    -2.) *
                (d2cb_c_o2_dpp_c_o22(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                        dcb_o2_dpp_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) +
                    dcb_c_o2_dpp_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                        d2cb_o2_dpp_o2dpp_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) -
                    d2cb_o2_dpp_c_o22(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) -
                    dcb_o2_dpp_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                        d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np)) *
                (dcb_c_o2_dpp_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                        (q_arcor_sys_np * (cb_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) -
                                              cb_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np))) -
                    dcb_o2_dpp_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                        (q_arcor_sys_np * (cb_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) -
                                              cb_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np)))) +
            (1. / V_vencor_sys_np) *
                pow((dcb_c_o2_dpp_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                            dcb_o2_dpp_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) -
                        dcb_o2_dpp_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                            dcb_c_o2_dpp_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np)),
                    -1.) *
                (d2cb_c_o2_dpp_c_o22(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                        (q_arcor_sys_np * (cb_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) -
                                              cb_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np))) +
                    dcb_c_o2_dpp_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) * q_arcor_sys_np *
                        dcb_o2_dpp_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) -
                    d2cb_o2_dpp_c_o22(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                        (q_arcor_sys_np * (cb_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) -
                                              cb_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np))) -
                    dcb_o2_dpp_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) * q_arcor_sys_np *
                        dcb_c_o2_dpp_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np)));
    // w.r.t. ppO2
    wkstiff(79, 79) =
        1. / ts_size +
        theta *
            (-(1. / V_vencor_sys_np) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                                dcb_o2_dpp_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) -
                            dcb_o2_dpp_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np)),
                        -2.) *
                    (d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                            dcb_o2_dpp_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) +
                        dcb_c_o2_dpp_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                            d2cb_o2_dpp_o22(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) -
                        d2cb_o2_dpp_o2dpp_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                            dcb_c_o2_dpp_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) -
                        dcb_o2_dpp_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                            d2cb_c_o2_dpp_o22(ppCO2_vencor_sys_np, ppO2_vencor_sys_np)) *
                    (dcb_c_o2_dpp_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                            (q_arcor_sys_np * (cb_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) -
                                                  cb_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np))) -
                        dcb_o2_dpp_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                            (q_arcor_sys_np *
                                (cb_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) -
                                    cb_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np)))) +
                (1. / V_vencor_sys_np) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                                dcb_o2_dpp_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) -
                            dcb_o2_dpp_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np)),
                        -1.) *
                    (d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                            (q_arcor_sys_np * (cb_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) -
                                                  cb_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np))) +
                        dcb_c_o2_dpp_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                            q_arcor_sys_np *
                            dcb_o2_dpp_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) -
                        d2cb_o2_dpp_o2dpp_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) *
                            (q_arcor_sys_np * (cb_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) -
                                                  cb_c_o2(ppCO2_arcor_sys_np, ppO2_arcor_sys_np))) -
                        dcb_o2_dpp_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) * q_arcor_sys_np *
                            dcb_c_o2_dpp_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np)));


    //////// systemic veins CO2
    // w.r.t. upstream flux - q_venspl_sys_np
    wkstiff(80, 13) =
        theta * ((1. / V_ven_sys_np) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                                dcb_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                            dcb_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np)),
                        -1.) *
                    (dcb_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                            (1.0 * (cb_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                                       cb_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np))) -
                        dcb_c_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                            (1.0 * (cb_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                                       cb_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np)))));
    // w.r.t. upstream flux - q_venespl_sys_np
    wkstiff(80, 15) =
        theta * ((1. / V_ven_sys_np) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                                dcb_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                            dcb_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np)),
                        -1.) *
                    (dcb_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                            (1.0 * (cb_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                                       cb_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np))) -
                        dcb_c_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                            (1.0 * (cb_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                                       cb_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np)))));
    // w.r.t. upstream flux - q_venmsc_sys_np
    wkstiff(80, 17) =
        theta * ((1. / V_ven_sys_np) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                                dcb_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                            dcb_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np)),
                        -1.) *
                    (dcb_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                            (1.0 * (cb_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                                       cb_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np))) -
                        dcb_c_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                            (1.0 * (cb_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                                       cb_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np)))));
    // w.r.t. upstream flux - q_vencer_sys_np
    wkstiff(80, 19) =
        theta * ((1. / V_ven_sys_np) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                                dcb_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                            dcb_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np)),
                        -1.) *
                    (dcb_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                            (1.0 * (cb_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                                       cb_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np))) -
                        dcb_c_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                            (1.0 * (cb_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                                       cb_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np)))));
    // w.r.t. upstream flux - q_vencor_sys_np
    wkstiff(80, 21) =
        theta * ((1. / V_ven_sys_np) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                                dcb_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                            dcb_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np)),
                        -1.) *
                    (dcb_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                            (1.0 * (cb_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                                       cb_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np))) -
                        dcb_c_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                            (1.0 * (cb_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                                       cb_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np)))));
    // w.r.t. mech. pressure
    wkstiff(80, 22) = theta * (c_ven_sys_ * (-1. / (V_ven_sys_np * V_ven_sys_np)) *
                                  pow((dcb_c_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                                              dcb_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                                          dcb_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                                              dcb_c_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np)),
                                      -1.) *
                                  (dcb_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                                          ((q_venspl_sys_np + q_venespl_sys_np + q_venmsc_sys_np +
                                               q_vencer_sys_np + q_vencor_sys_np) *
                                                  cb_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                                              cb_c_o2(ppCO2_at_r_np, ppO2_at_r_np)) -
                                      dcb_c_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                                          ((q_venspl_sys_np + q_venespl_sys_np + q_venmsc_sys_np +
                                               q_vencer_sys_np + q_vencor_sys_np) *
                                                  cb_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                                              cb_o2(ppCO2_at_r_np, ppO2_at_r_np))));
    // w.r.t. upstream ppCO2_venspl_sys
    wkstiff(80, 70) =
        theta *
        (-(1. / V_ven_sys_np) *
            pow((dcb_c_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                        dcb_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                    dcb_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np)),
                -1.) *
            (dcb_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                    (q_venspl_sys_np)*dcb_c_o2_dpp_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) -
                dcb_c_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                    (q_venspl_sys_np)*dcb_o2_dpp_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np)));
    // w.r.t. upstream ppO2_venspl_sys
    wkstiff(80, 71) =
        theta *
        (-(1. / V_ven_sys_np) *
            pow((dcb_c_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                        dcb_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                    dcb_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np)),
                -1.) *
            (dcb_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                    (q_venspl_sys_np)*dcb_c_o2_dpp_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) -
                dcb_c_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                    (q_venspl_sys_np)*dcb_o2_dpp_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np)));
    // w.r.t. upstream ppCO2_venespl_sys
    wkstiff(80, 72) =
        theta *
        (-(1. / V_ven_sys_np) *
            pow((dcb_c_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                        dcb_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                    dcb_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np)),
                -1.) *
            (dcb_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                    (q_venespl_sys_np)*dcb_c_o2_dpp_c_o2(
                        ppCO2_venespl_sys_np, ppO2_venespl_sys_np) -
                dcb_c_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                    (q_venespl_sys_np)*dcb_o2_dpp_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np)));
    // w.r.t. upstream ppO2_venespl_sys
    wkstiff(80, 73) =
        theta *
        (-(1. / V_ven_sys_np) *
            pow((dcb_c_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                        dcb_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                    dcb_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np)),
                -1.) *
            (dcb_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                    (q_venespl_sys_np)*dcb_c_o2_dpp_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) -
                dcb_c_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                    (q_venespl_sys_np)*dcb_o2_dpp_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np)));
    // w.r.t. upstream ppCO2_venmsc_sys
    wkstiff(80, 74) =
        theta *
        (-(1. / V_ven_sys_np) *
            pow((dcb_c_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                        dcb_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                    dcb_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np)),
                -1.) *
            (dcb_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                    (q_venmsc_sys_np)*dcb_c_o2_dpp_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) -
                dcb_c_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                    (q_venmsc_sys_np)*dcb_o2_dpp_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np)));
    // w.r.t. upstream ppO2_venmsc_sys
    wkstiff(80, 75) =
        theta *
        (-(1. / V_ven_sys_np) *
            pow((dcb_c_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                        dcb_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                    dcb_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np)),
                -1.) *
            (dcb_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                    (q_venmsc_sys_np)*dcb_c_o2_dpp_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) -
                dcb_c_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                    (q_venmsc_sys_np)*dcb_o2_dpp_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np)));
    // w.r.t. upstream ppCO2_vencer_sys
    wkstiff(80, 76) =
        theta *
        (-(1. / V_ven_sys_np) *
            pow((dcb_c_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                        dcb_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                    dcb_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np)),
                -1.) *
            (dcb_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                    (q_vencer_sys_np)*dcb_c_o2_dpp_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) -
                dcb_c_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                    (q_vencer_sys_np)*dcb_o2_dpp_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np)));
    // w.r.t. upstream ppO2_vencer_sys
    wkstiff(80, 77) =
        theta *
        (-(1. / V_ven_sys_np) *
            pow((dcb_c_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                        dcb_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                    dcb_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np)),
                -1.) *
            (dcb_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                    (q_vencer_sys_np)*dcb_c_o2_dpp_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) -
                dcb_c_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                    (q_vencer_sys_np)*dcb_o2_dpp_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np)));
    // w.r.t. upstream ppCO2_vencor_sys
    wkstiff(80, 78) =
        theta *
        (-(1. / V_ven_sys_np) *
            pow((dcb_c_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                        dcb_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                    dcb_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np)),
                -1.) *
            (dcb_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                    (q_vencor_sys_np)*dcb_c_o2_dpp_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) -
                dcb_c_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                    (q_vencor_sys_np)*dcb_o2_dpp_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np)));
    // w.r.t. upstream ppO2_vencor_sys
    wkstiff(80, 79) =
        theta *
        (-(1. / V_ven_sys_np) *
            pow((dcb_c_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                        dcb_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                    dcb_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np)),
                -1.) *
            (dcb_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                    (q_vencor_sys_np)*dcb_c_o2_dpp_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) -
                dcb_c_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                    (q_vencor_sys_np)*dcb_o2_dpp_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np)));
    // w.r.t. ppCO2
    wkstiff(80, 80) =
        1. / ts_size +
        theta *
            (-(1. / V_ven_sys_np) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                                dcb_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                            dcb_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np)),
                        -2.) *
                    (d2cb_c_o2_dpp_c_o22(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                            dcb_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) +
                        dcb_c_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                            d2cb_o2_dpp_o2dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                        d2cb_o2_dpp_c_o22(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                            dcb_c_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                        dcb_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                            d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np)) *
                    (dcb_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                            ((q_venspl_sys_np + q_venespl_sys_np + q_venmsc_sys_np +
                                 q_vencer_sys_np + q_vencor_sys_np) *
                                    cb_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                                (q_venspl_sys_np *
                                        cb_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) +
                                    q_venespl_sys_np *
                                        cb_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) +
                                    q_venmsc_sys_np *
                                        cb_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) +
                                    q_vencer_sys_np *
                                        cb_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) +
                                    q_vencor_sys_np *
                                        cb_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np))) -
                        dcb_c_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                            ((q_venspl_sys_np + q_venespl_sys_np + q_venmsc_sys_np +
                                 q_vencer_sys_np + q_vencor_sys_np) *
                                (cb_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                                    (q_venspl_sys_np *
                                            cb_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) +
                                        q_venespl_sys_np *
                                            cb_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) +
                                        q_venmsc_sys_np *
                                            cb_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) +
                                        q_vencer_sys_np *
                                            cb_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) +
                                        q_vencor_sys_np *
                                            cb_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np))))) +
                (1. / V_ven_sys_np) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                                dcb_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                            dcb_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np)),
                        -1.) *
                    (d2cb_o2_dpp_o2dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                            ((q_venspl_sys_np + q_venespl_sys_np + q_venmsc_sys_np +
                                 q_vencer_sys_np + q_vencor_sys_np) *
                                    cb_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                                (q_venspl_sys_np *
                                        cb_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) +
                                    q_venespl_sys_np *
                                        cb_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) +
                                    q_venmsc_sys_np *
                                        cb_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) +
                                    q_vencer_sys_np *
                                        cb_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) +
                                    q_vencor_sys_np *
                                        cb_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np))) +
                        dcb_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                            (q_venspl_sys_np + q_venespl_sys_np + q_venmsc_sys_np +
                                q_vencer_sys_np + q_vencor_sys_np) *
                            dcb_c_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                        d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                            ((q_venspl_sys_np + q_venespl_sys_np + q_venmsc_sys_np +
                                 q_vencer_sys_np + q_vencor_sys_np) *
                                    cb_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                                (q_venspl_sys_np * cb_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) +
                                    q_venespl_sys_np *
                                        cb_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) +
                                    q_venmsc_sys_np *
                                        cb_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) +
                                    q_vencer_sys_np *
                                        cb_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) +
                                    q_vencor_sys_np *
                                        cb_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np))) -
                        dcb_c_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                            (q_venspl_sys_np + q_venespl_sys_np + q_venmsc_sys_np +
                                q_vencer_sys_np + q_vencor_sys_np) *
                            dcb_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np)));
    // w.r.t. ppO2
    wkstiff(80, 81) =
        theta *
        (-(1. / V_ven_sys_np) *
                pow((dcb_c_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                            dcb_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                        dcb_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                            dcb_c_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np)),
                    -2.) *
                (d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                        dcb_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) +
                    dcb_c_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                        d2cb_o2_dpp_o22(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                    d2cb_o2_dpp_o2dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                    dcb_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                        d2cb_c_o2_dpp_o22(ppCO2_ven_sys_np, ppO2_ven_sys_np)) *
                (dcb_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                        ((q_venspl_sys_np + q_venespl_sys_np + q_venmsc_sys_np + q_vencer_sys_np +
                             q_vencor_sys_np) *
                                cb_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                            (q_venspl_sys_np * cb_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) +
                                q_venespl_sys_np *
                                    cb_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) +
                                q_venmsc_sys_np * cb_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) +
                                q_vencer_sys_np * cb_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) +
                                q_vencor_sys_np *
                                    cb_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np))) -
                    dcb_c_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                        ((q_venspl_sys_np + q_venespl_sys_np + q_venmsc_sys_np + q_vencer_sys_np +
                             q_vencor_sys_np) *
                            (cb_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                                (q_venspl_sys_np * cb_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) +
                                    q_venespl_sys_np *
                                        cb_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) +
                                    q_venmsc_sys_np *
                                        cb_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) +
                                    q_vencer_sys_np *
                                        cb_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) +
                                    q_vencor_sys_np *
                                        cb_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np))))) +
            (1. / V_ven_sys_np) *
                pow((dcb_c_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                            dcb_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                        dcb_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                            dcb_c_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np)),
                    -1.) *
                (d2cb_o2_dpp_o22(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                        ((q_venspl_sys_np + q_venespl_sys_np + q_venmsc_sys_np + q_vencer_sys_np +
                             q_vencor_sys_np) *
                                cb_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                            (q_venspl_sys_np * cb_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) +
                                q_venespl_sys_np *
                                    cb_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) +
                                q_venmsc_sys_np * cb_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) +
                                q_vencer_sys_np * cb_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) +
                                q_vencor_sys_np *
                                    cb_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np))) +
                    dcb_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                        (q_venspl_sys_np + q_venespl_sys_np + q_venmsc_sys_np + q_vencer_sys_np +
                            q_vencor_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                    d2cb_c_o2_dpp_o22(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                        ((q_venspl_sys_np + q_venespl_sys_np + q_venmsc_sys_np + q_vencer_sys_np +
                             q_vencor_sys_np) *
                                cb_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                            (q_venspl_sys_np * cb_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) +
                                q_venespl_sys_np *
                                    cb_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) +
                                q_venmsc_sys_np * cb_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) +
                                q_vencer_sys_np * cb_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) +
                                q_vencor_sys_np * cb_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np))) -
                    dcb_c_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                        (q_venspl_sys_np + q_venespl_sys_np + q_venmsc_sys_np + q_vencer_sys_np +
                            q_vencor_sys_np) *
                        dcb_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np)));

    //////// systemic veins O2
    // w.r.t. upstream flux - q_venspl_sys_np
    wkstiff(81, 13) =
        theta * ((1. / V_ven_sys_np) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                                dcb_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                            dcb_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np)),
                        -1.) *
                    (dcb_c_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                            (1.0 * (cb_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                                       cb_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np))) -
                        dcb_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                            (1.0 * (cb_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                                       cb_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np)))));
    // w.r.t. upstream flux - q_venespl_sys_np
    wkstiff(81, 15) =
        theta * ((1. / V_ven_sys_np) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                                dcb_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                            dcb_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np)),
                        -1.) *
                    (dcb_c_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                            (1.0 * (cb_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                                       cb_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np))) -
                        dcb_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                            (1.0 * (cb_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                                       cb_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np)))));
    // w.r.t. upstream flux - q_venmsc_sys_np
    wkstiff(81, 17) =
        theta * ((1. / V_ven_sys_np) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                                dcb_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                            dcb_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np)),
                        -1.) *
                    (dcb_c_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                            (1.0 * (cb_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                                       cb_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np))) -
                        dcb_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                            (1.0 * (cb_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                                       cb_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np)))));
    // w.r.t. upstream flux - q_vencer_sys_np
    wkstiff(81, 19) =
        theta * ((1. / V_ven_sys_np) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                                dcb_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                            dcb_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np)),
                        -1.) *
                    (dcb_c_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                            (1.0 * (cb_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                                       cb_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np))) -
                        dcb_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                            (1.0 * (cb_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                                       cb_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np)))));
    // w.r.t. upstream flux - q_vencor_sys_np
    wkstiff(81, 21) =
        theta * ((1. / V_ven_sys_np) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                                dcb_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                            dcb_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np)),
                        -1.) *
                    (dcb_c_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                            (1.0 * (cb_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                                       cb_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np))) -
                        dcb_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                            (1.0 * (cb_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                                       cb_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np)))));
    // w.r.t. mech. pressure
    wkstiff(81, 22) = theta * (c_ven_sys_ * (-1. / (V_ven_sys_np * V_ven_sys_np)) *
                                  pow((dcb_c_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                                              dcb_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                                          dcb_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                                              dcb_c_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np)),
                                      -1.) *
                                  (dcb_c_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                                          ((q_venspl_sys_np + q_venespl_sys_np + q_venmsc_sys_np +
                                               q_vencer_sys_np + q_vencor_sys_np) *
                                                  cb_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                                              cb_o2(ppCO2_at_r_np, ppO2_at_r_np)) -
                                      dcb_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                                          ((q_venspl_sys_np + q_venespl_sys_np + q_venmsc_sys_np +
                                               q_vencer_sys_np + q_vencor_sys_np) *
                                                  cb_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                                              cb_c_o2(ppCO2_at_r_np, ppO2_at_r_np))));
    // w.r.t. upstream ppCO2_venspl_sys
    wkstiff(81, 70) =
        theta *
        (-(1. / V_ven_sys_np) *
            pow((dcb_c_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                        dcb_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                    dcb_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np)),
                -1.) *
            (dcb_c_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                    (q_venspl_sys_np)*dcb_o2_dpp_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) -
                dcb_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                    (q_venspl_sys_np)*dcb_c_o2_dpp_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np)));
    // w.r.t. upstream ppO2_venspl_sys
    wkstiff(81, 71) =
        theta *
        (-(1. / V_ven_sys_np) *
            pow((dcb_c_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                        dcb_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                    dcb_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np)),
                -1.) *
            (dcb_c_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                    (q_venspl_sys_np)*dcb_o2_dpp_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) -
                dcb_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                    (q_venspl_sys_np)*dcb_c_o2_dpp_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np)));
    // w.r.t. upstream ppCO2_venespl_sys
    wkstiff(81, 72) =
        theta *
        (-(1. / V_ven_sys_np) *
            pow((dcb_c_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                        dcb_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                    dcb_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np)),
                -1.) *
            (dcb_c_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                    (q_venespl_sys_np)*dcb_o2_dpp_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) -
                dcb_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                    (q_venespl_sys_np)*dcb_c_o2_dpp_c_o2(
                        ppCO2_venespl_sys_np, ppO2_venespl_sys_np)));
    // w.r.t. upstream ppO2_venespl_sys
    wkstiff(81, 73) =
        theta *
        (-(1. / V_ven_sys_np) *
            pow((dcb_c_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                        dcb_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                    dcb_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np)),
                -1.) *
            (dcb_c_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                    (q_venespl_sys_np)*dcb_o2_dpp_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) -
                dcb_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                    (q_venespl_sys_np)*dcb_c_o2_dpp_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np)));
    // w.r.t. upstream ppCO2_venmsc_sys
    wkstiff(81, 74) =
        theta *
        (-(1. / V_ven_sys_np) *
            pow((dcb_c_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                        dcb_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                    dcb_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np)),
                -1.) *
            (dcb_c_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                    (q_venmsc_sys_np)*dcb_o2_dpp_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) -
                dcb_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                    (q_venmsc_sys_np)*dcb_c_o2_dpp_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np)));
    // w.r.t. upstream ppO2_venmsc_sys
    wkstiff(81, 75) =
        theta *
        (-(1. / V_ven_sys_np) *
            pow((dcb_c_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                        dcb_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                    dcb_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np)),
                -1.) *
            (dcb_c_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                    (q_venmsc_sys_np)*dcb_o2_dpp_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) -
                dcb_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                    (q_venmsc_sys_np)*dcb_c_o2_dpp_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np)));
    // w.r.t. upstream ppCO2_vencer_sys
    wkstiff(81, 76) =
        theta *
        (-(1. / V_ven_sys_np) *
            pow((dcb_c_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                        dcb_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                    dcb_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np)),
                -1.) *
            (dcb_c_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                    (q_vencer_sys_np)*dcb_o2_dpp_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) -
                dcb_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                    (q_vencer_sys_np)*dcb_c_o2_dpp_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np)));
    // w.r.t. upstream ppO2_vencer_sys
    wkstiff(81, 77) =
        theta *
        (-(1. / V_ven_sys_np) *
            pow((dcb_c_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                        dcb_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                    dcb_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np)),
                -1.) *
            (dcb_c_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                    (q_vencer_sys_np)*dcb_o2_dpp_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) -
                dcb_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                    (q_vencer_sys_np)*dcb_c_o2_dpp_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np)));
    // w.r.t. upstream ppCO2_vencor_sys
    wkstiff(81, 78) =
        theta *
        (-(1. / V_ven_sys_np) *
            pow((dcb_c_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                        dcb_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                    dcb_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np)),
                -1.) *
            (dcb_c_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                    (q_vencor_sys_np)*dcb_o2_dpp_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) -
                dcb_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                    (q_vencor_sys_np)*dcb_c_o2_dpp_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np)));
    // w.r.t. upstream ppO2_vencor_sys
    wkstiff(81, 79) =
        theta *
        (-(1. / V_ven_sys_np) *
            pow((dcb_c_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                        dcb_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                    dcb_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np)),
                -1.) *
            (dcb_c_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                    (q_vencor_sys_np)*dcb_o2_dpp_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np) -
                dcb_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                    (q_vencor_sys_np)*dcb_c_o2_dpp_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np)));
    // w.r.t. ppCO2
    wkstiff(81, 80) =
        theta *
        (-(1. / V_ven_sys_np) *
                pow((dcb_c_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                            dcb_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                        dcb_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                            dcb_c_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np)),
                    -2.) *
                (d2cb_c_o2_dpp_c_o22(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                        dcb_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) +
                    dcb_c_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                        d2cb_o2_dpp_o2dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                    d2cb_o2_dpp_c_o22(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                        dcb_c_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                    dcb_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                        d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np)) *
                (dcb_c_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                        ((q_venspl_sys_np + q_venespl_sys_np + q_venmsc_sys_np + q_vencer_sys_np +
                             q_vencor_sys_np) *
                                cb_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                            (q_venspl_sys_np * cb_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) +
                                q_venespl_sys_np *
                                    cb_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) +
                                q_venmsc_sys_np * cb_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) +
                                q_vencer_sys_np * cb_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) +
                                q_vencor_sys_np * cb_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np))) -
                    dcb_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                        ((q_venspl_sys_np + q_venespl_sys_np + q_venmsc_sys_np + q_vencer_sys_np +
                             q_vencor_sys_np) *
                                cb_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                            (q_venspl_sys_np * cb_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) +
                                q_venespl_sys_np *
                                    cb_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) +
                                q_venmsc_sys_np * cb_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) +
                                q_vencer_sys_np * cb_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) +
                                q_vencor_sys_np *
                                    cb_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np)))) +
            (1. / V_ven_sys_np) *
                pow((dcb_c_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                            dcb_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                        dcb_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                            dcb_c_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np)),
                    -1.) *
                (d2cb_c_o2_dpp_c_o22(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                        ((q_venspl_sys_np + q_venespl_sys_np + q_venmsc_sys_np + q_vencer_sys_np +
                             q_vencor_sys_np) *
                                cb_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                            (q_venspl_sys_np * cb_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) +
                                q_venespl_sys_np *
                                    cb_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) +
                                q_venmsc_sys_np * cb_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) +
                                q_vencer_sys_np * cb_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) +
                                q_vencor_sys_np * cb_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np))) +
                    dcb_c_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                        (q_venspl_sys_np + q_venespl_sys_np + q_venmsc_sys_np + q_vencer_sys_np +
                            q_vencor_sys_np) *
                        dcb_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                    d2cb_o2_dpp_c_o22(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                        ((q_venspl_sys_np + q_venespl_sys_np + q_venmsc_sys_np + q_vencer_sys_np +
                             q_vencor_sys_np) *
                                cb_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                            (q_venspl_sys_np * cb_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) +
                                q_venespl_sys_np *
                                    cb_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) +
                                q_venmsc_sys_np * cb_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) +
                                q_vencer_sys_np * cb_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) +
                                q_vencor_sys_np *
                                    cb_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np))) -
                    dcb_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                        (q_venspl_sys_np + q_venespl_sys_np + q_venmsc_sys_np + q_vencer_sys_np +
                            q_vencor_sys_np) *
                        dcb_c_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np)));
    // w.r.t. ppO2
    wkstiff(81, 81) =
        1. / ts_size +
        theta *
            (-(1. / V_ven_sys_np) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                                dcb_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                            dcb_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np)),
                        -2.) *
                    (d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                            dcb_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) +
                        dcb_c_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                            d2cb_o2_dpp_o22(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                        d2cb_o2_dpp_o2dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                            dcb_c_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                        dcb_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                            d2cb_c_o2_dpp_o22(ppCO2_ven_sys_np, ppO2_ven_sys_np)) *
                    (dcb_c_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                            ((q_venspl_sys_np + q_venespl_sys_np + q_venmsc_sys_np +
                                 q_vencer_sys_np + q_vencor_sys_np) *
                                    cb_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                                (q_venspl_sys_np * cb_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) +
                                    q_venespl_sys_np *
                                        cb_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) +
                                    q_venmsc_sys_np *
                                        cb_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) +
                                    q_vencer_sys_np *
                                        cb_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) +
                                    q_vencor_sys_np *
                                        cb_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np))) -
                        dcb_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                            ((q_venspl_sys_np + q_venespl_sys_np + q_venmsc_sys_np +
                                 q_vencer_sys_np + q_vencor_sys_np) *
                                    cb_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                                (q_venspl_sys_np *
                                        cb_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) +
                                    q_venespl_sys_np *
                                        cb_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) +
                                    q_venmsc_sys_np *
                                        cb_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) +
                                    q_vencer_sys_np *
                                        cb_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) +
                                    q_vencor_sys_np *
                                        cb_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np)))) +
                (1. / V_ven_sys_np) *
                    pow((dcb_c_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                                dcb_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                            dcb_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                                dcb_c_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np)),
                        -1.) *
                    (d2cb_c_o2_dpp_c_o2dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                            ((q_venspl_sys_np + q_venespl_sys_np + q_venmsc_sys_np +
                                 q_vencer_sys_np + q_vencor_sys_np) *
                                    cb_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                                (q_venspl_sys_np * cb_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) +
                                    q_venespl_sys_np *
                                        cb_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) +
                                    q_venmsc_sys_np *
                                        cb_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) +
                                    q_vencer_sys_np *
                                        cb_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) +
                                    q_vencor_sys_np *
                                        cb_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np))) +
                        dcb_c_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                            (q_venspl_sys_np + q_venespl_sys_np + q_venmsc_sys_np +
                                q_vencer_sys_np + q_vencor_sys_np) *
                            dcb_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                        d2cb_o2_dpp_o2dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                            ((q_venspl_sys_np + q_venespl_sys_np + q_venmsc_sys_np +
                                 q_vencer_sys_np + q_vencor_sys_np) *
                                    cb_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) -
                                (q_venspl_sys_np *
                                        cb_c_o2(ppCO2_venspl_sys_np, ppO2_venspl_sys_np) +
                                    q_venespl_sys_np *
                                        cb_c_o2(ppCO2_venespl_sys_np, ppO2_venespl_sys_np) +
                                    q_venmsc_sys_np *
                                        cb_c_o2(ppCO2_venmsc_sys_np, ppO2_venmsc_sys_np) +
                                    q_vencer_sys_np *
                                        cb_c_o2(ppCO2_vencer_sys_np, ppO2_vencer_sys_np) +
                                    q_vencor_sys_np *
                                        cb_c_o2(ppCO2_vencor_sys_np, ppO2_vencor_sys_np))) -
                        dcb_o2_dpp_c_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np) *
                            (q_venspl_sys_np + q_venespl_sys_np + q_venmsc_sys_np +
                                q_vencer_sys_np + q_vencor_sys_np) *
                            dcb_c_o2_dpp_o2(ppCO2_ven_sys_np, ppO2_ven_sys_np)));
  }

  return;
}

// cbO2 and its derivatives
double Utils::CardiovascularRespiratory0DSysPulPeriphCirculation::cb_o2(double ppCO2, double ppO2)
{
  //  const double n = 2.7;
  //  const double ppO2_50 = 26.8/7.500615; // 26.8 mmHg -> convert to kPa!
  // with Hill oxygen dissociation curve - simplest form, independent of CO2 and pH !
  const double cbO2_val = alpha_o2_ * ppO2 + c_hb_ * s_o2(ppCO2, ppO2);

  return cbO2_val;
}
double Utils::CardiovascularRespiratory0DSysPulPeriphCirculation::s_o2(double ppCO2, double ppO2)
{
  const double n = 2.7;
  const double ppO2_50 = 26.8 / 7.500615;  // 26.8 mmHg -> convert to kPa!
  // with Hill oxygen dissociation curve - simplest form, independent of CO2 and pH !

  const double SO2_val = pow((ppO2 / ppO2_50), n) / (1. + pow((ppO2 / ppO2_50), n));

  return SO2_val;
}
// w.r.t. O2
double Utils::CardiovascularRespiratory0DSysPulPeriphCirculation::dcb_o2_dpp_o2(
    double ppCO2, double ppO2)
{
  const double n = 2.7;
  const double ppO2_50 = 26.8 / 7.500615;  // 26.8 mmHg -> convert to kPa!

  const double dcbO2_dppO2_val = alpha_o2_ + c_hb_ * n * pow((ppO2 / ppO2_50), n) /
                                                 (pow((1. + pow((ppO2 / ppO2_50), n)), 2.) * ppO2);

  return dcbO2_dppO2_val;
}
double Utils::CardiovascularRespiratory0DSysPulPeriphCirculation::d2cb_o2_dpp_o22(
    double ppCO2, double ppO2)
{
  const double n = 2.7;
  const double ppO2_50 = 26.8 / 7.500615;  // 26.8 mmHg -> convert to kPa!

  const double d2cbO2_dppO22_val =
      c_hb_ *
      (pow((ppO2 / ppO2_50), n) * n - pow((ppO2 / ppO2_50), (2. * n)) * n -
          pow((ppO2 / ppO2_50), n) - pow((ppO2 / ppO2_50), (2. * n))) *
      n / (pow((1. + pow((ppO2 / ppO2_50), n)), 3.) * ppO2 * ppO2);

  return d2cbO2_dppO22_val;
}
// w.r.t. CO2
double Utils::CardiovascularRespiratory0DSysPulPeriphCirculation::dcb_o2_dpp_c_o2(
    double ppCO2, double ppO2)
{
  const double dcbO2_dppCO2_val = 0.;

  return dcbO2_dppCO2_val;
}
double Utils::CardiovascularRespiratory0DSysPulPeriphCirculation::d2cb_o2_dpp_c_o22(
    double ppCO2, double ppO2)
{
  const double d2cbO2_dppCO22_val = 0.;

  return d2cbO2_dppCO22_val;
}
double Utils::CardiovascularRespiratory0DSysPulPeriphCirculation::d2cb_o2_dpp_o2dpp_c_o2(
    double ppCO2, double ppO2)
{
  const double d2cbO2_dppO2dppCO2_val = 0.;

  return d2cbO2_dppO2dppCO2_val;
}


// cbCO2 and its derivatives
double Utils::CardiovascularRespiratory0DSysPulPeriphCirculation::cb_c_o2(double ppCO2, double ppO2)
{
  const double cbCO2_val = alpha_c_o2_ * ppCO2;

  return cbCO2_val;
}
// w.r.t. CO2
double Utils::CardiovascularRespiratory0DSysPulPeriphCirculation::dcb_c_o2_dpp_c_o2(
    double ppCO2, double ppO2)
{
  const double dcbCO2_dppCO2_val = alpha_c_o2_;

  return dcbCO2_dppCO2_val;
}
double Utils::CardiovascularRespiratory0DSysPulPeriphCirculation::d2cb_c_o2_dpp_c_o22(
    double ppCO2, double ppO2)
{
  const double d2cbCO2_dppCO22_val = 0.;

  return d2cbCO2_dppCO22_val;
}
// w.r.t. O2
double Utils::CardiovascularRespiratory0DSysPulPeriphCirculation::dcb_c_o2_dpp_o2(
    double ppCO2, double ppO2)
{
  const double dcbCO2_dppO2_val = 0.;

  return dcbCO2_dppO2_val;
}
double Utils::CardiovascularRespiratory0DSysPulPeriphCirculation::d2cb_c_o2_dpp_o22(
    double ppCO2, double ppO2)
{
  const double d2cbCO2_dppO22_val = 0.;

  return d2cbCO2_dppO22_val;
}
double Utils::CardiovascularRespiratory0DSysPulPeriphCirculation::d2cb_c_o2_dpp_c_o2dpp_o2(
    double ppCO2, double ppO2)
{
  const double d2cbCO2_dppCO2dppO2_val = 0.;

  return d2cbCO2_dppCO2dppO2_val;
}



double Utils::CardiovascularRespiratory0DSysPulPeriphCirculation::ct_o2(double ppO2)
{
  const double ctO2_val = alpha_o2_ * ppO2;

  return ctO2_val;
}

double Utils::CardiovascularRespiratory0DSysPulPeriphCirculation::dct_o2_dpp_o2(double ppO2)
{
  const double dctO2_dppO2_val = alpha_o2_;

  return dctO2_dppO2_val;
}

double Utils::CardiovascularRespiratory0DSysPulPeriphCirculation::d2ct_o2_dpp_o22(double ppO2)
{
  const double d2ctO2_dppO22_val = 0.;

  return d2ctO2_dppO22_val;
}



double Utils::CardiovascularRespiratory0DSysPulPeriphCirculation::ct_c_o2(double ppCO2)
{
  const double ctCO2_val = alpha_c_o2_ * ppCO2;

  return ctCO2_val;
}

double Utils::CardiovascularRespiratory0DSysPulPeriphCirculation::dct_c_o2_dpp_c_o2(double ppCO2)
{
  const double dctCO2_dppCO2_val = alpha_c_o2_;

  return dctCO2_dppCO2_val;
}

double Utils::CardiovascularRespiratory0DSysPulPeriphCirculation::d2ct_c_o2_dpp_c_o22(double ppCO2)
{
  const double d2ctCO2_dppCO22_val = 0.;

  return d2ctCO2_dppCO22_val;
}


// double Utils::CardiovascularRespiratory0DSysPulPeriphCirculation::y(double ppCO2,double ppO2)
//{
//  const double T_blood = 37.; // in  degrees celsius !
//  // ppO2 in log10 in kPa!!
//  const double y_val = 1.875 + log10(ppO2) - (1.946 + a(ppCO2,ppO2) + 0.055(T_blood-37.));
//
//  return y_val;
//}
//
// double Utils::CardiovascularRespiratory0DSysPulPeriphCirculation::a(double ppCO2,double ppO2)
//{
//  const double x_Hbf = 1.; // fraction of fetal hemoglobin concentration in the blood
//  const double x_HbCO = 1.; // fraction of carboxyhemoglobin in the blood
//  const double x_Hi = 1.; // fraction of glycohemoglobin in the blood
//  const double c_DPG = 1.; // concentration of 2,3-diphosphogycerate in the erythrocyte, in mol/l
//  // ppCO2 in log10 in kPa!!
//  const double a_val = -0.72*(pH(...)) + 0.09*log10(ppCO2/5.33) + (0.07-0.03*x_Hbf)*(c_DPG-5) -
//  0.368*x_HbCO-0.174*x_Hi-0.28*x_Hbf;
//
//  return a_val;
//}



/*-----------------------------------------------------------------------*
 *-----------------------------------------------------------------------*/
void Utils::CardiovascularRespiratory0DSysPulPeriphCirculation::initialize(
    Teuchos::ParameterList& params, std::shared_ptr<Core::LinAlg::Vector<double>> sysvec1,
    std::shared_ptr<Core::LinAlg::Vector<double>> sysvec2)
{
  if (!(actdisc_->filled())) FOUR_C_THROW("fill_complete() was not called");
  if (!actdisc_->have_dofs()) FOUR_C_THROW("assign_degrees_of_freedom() was not called");
  // get the current time
  // const double time = params.get("total time",-1.0);

  params.set("action", "calc_struct_constrvol");

  const bool assvec1 = sysvec1 != nullptr;

  // global and local ID of this bc in the redundant vectors
  const int offsetID = params.get<int>("OffsetID");
  std::vector<int> gindex(num_dof_);
  gindex[0] = offsetID;
  for (int j = 1; j < num_dof_; j++) gindex[j] = gindex[0] + j;

  std::vector<double> initvals(num_dof_);

  Teuchos::ParameterList artvensyspulpar =
      Global::Problem::instance()->cardiovascular0_d_structural_params().sublist(
          "SYS-PUL CIRCULATION PARAMETERS");

  Teuchos::ParameterList respirpar =
      Global::Problem::instance()->cardiovascular0_d_structural_params().sublist(
          "RESPIRATORY PARAMETERS");

  initvals[0] = artvensyspulpar.get("p_at_l_0", 0.0);
  initvals[1] = artvensyspulpar.get("q_vin_l_0", 0.0);
  initvals[2] = artvensyspulpar.get("q_vout_l_0", 0.0);
  initvals[3] = artvensyspulpar.get("p_v_l_0", 0.0);
  initvals[4] = artvensyspulpar.get("p_ar_sys_0", 0.0);
  initvals[5] = artvensyspulpar.get("q_ar_sys_0", 0.0);

  initvals[6] = artvensyspulpar.get("p_arperi_sys_0", 0.0);
  initvals[7] = artvensyspulpar.get("q_arspl_sys_0", 0.0);
  initvals[8] = artvensyspulpar.get("q_arespl_sys_0", 0.0);
  initvals[9] = artvensyspulpar.get("q_armsc_sys_0", 0.0);
  initvals[10] = artvensyspulpar.get("q_arcer_sys_0", 0.0);
  initvals[11] = artvensyspulpar.get("q_arcor_sys_0", 0.0);
  initvals[12] = artvensyspulpar.get("p_venspl_sys_0", 0.0);
  initvals[13] = artvensyspulpar.get("q_venspl_sys_0", 0.0);
  initvals[14] = artvensyspulpar.get("p_venespl_sys_0", 0.0);
  initvals[15] = artvensyspulpar.get("q_venespl_sys_0", 0.0);
  initvals[16] = artvensyspulpar.get("p_venmsc_sys_0", 0.0);
  initvals[17] = artvensyspulpar.get("q_venmsc_sys_0", 0.0);
  initvals[18] = artvensyspulpar.get("p_vencer_sys_0", 0.0);
  initvals[19] = artvensyspulpar.get("q_vencer_sys_0", 0.0);
  initvals[20] = artvensyspulpar.get("p_vencor_sys_0", 0.0);
  initvals[21] = artvensyspulpar.get("q_vencor_sys_0", 0.0);

  initvals[22] = artvensyspulpar.get("p_ven_sys_0", 0.0);
  initvals[23] = artvensyspulpar.get("q_ven_sys_0", 0.0);
  initvals[24] = artvensyspulpar.get("p_at_r_0", 0.0);
  initvals[25] = artvensyspulpar.get("q_vin_r_0", 0.0);
  initvals[26] = artvensyspulpar.get("q_vout_r_0", 0.0);
  initvals[27] = artvensyspulpar.get("p_v_r_0", 0.0);
  initvals[28] = artvensyspulpar.get("p_ar_pul_0", 0.0);
  initvals[29] = artvensyspulpar.get("q_ar_pul_0", 0.0);
  initvals[30] = artvensyspulpar.get("p_cap_pul_0", 0.0);
  initvals[31] = artvensyspulpar.get("q_cap_pul_0", 0.0);
  initvals[32] = artvensyspulpar.get("p_ven_pul_0", 0.0);
  initvals[33] = artvensyspulpar.get("q_ven_pul_0", 0.0);

  switch (respiratory_model_)
  {
    case Inpar::Cardiovascular0D::resp_none:
      break;
    case Inpar::Cardiovascular0D::resp_standard:

      // initial value of time-varying pleural pressure
      double U_t_0 = 0.0;
      if (u_t_curve_ >= 0)
        U_t_0 = Global::Problem::instance()
                    ->function_by_id<Core::Utils::FunctionOfTime>(u_t_curve_)
                    .evaluate(0);

      double V_alv_0 = respirpar.get("V_alv_0", -1.0);
      if (V_alv_0 >= 0) initvals[34] = V_alv_0;
      if (V_alv_0 < 0) initvals[34] = (u_m_ - U_t_0) / e_alv_ + v_lung_u_;

      initvals[35] = respirpar.get("q_alv_0", 0.0);

      double p_alv_0 = respirpar.get("p_alv_0", -1.0);
      if (p_alv_0 >= 0) initvals[36] = p_alv_0;
      if (p_alv_0 < 0) initvals[36] = u_m_;

      initvals[37] = respirpar.get("fCO2_alv_0", 0.05263);
      initvals[38] = respirpar.get("fO2_alv_0", 0.1368);
      initvals[39] = respirpar.get("q_arspl_sys_in_0", 0.0);
      initvals[40] = respirpar.get("q_arespl_sys_in_0", 0.0);
      initvals[41] = respirpar.get("q_armsc_sys_in_0", 0.0);
      initvals[42] = respirpar.get("q_arcer_sys_in_0", 0.0);
      initvals[43] = respirpar.get("q_arcor_sys_in_0", 0.0);
      initvals[44] = respirpar.get("ppCO2_at_r_0", 5.0);
      initvals[45] = respirpar.get("ppO2_at_r_0", 10.0);
      initvals[46] = respirpar.get("ppCO2_v_r_0", 5.0);
      initvals[47] = respirpar.get("ppO2_v_r_0", 10.0);
      initvals[48] = respirpar.get("ppCO2_ar_pul_0", 5.0);
      initvals[49] = respirpar.get("ppO2_ar_pul_0", 10.0);
      initvals[50] = respirpar.get("ppCO2_cap_pul_0", 5.0);
      initvals[51] = respirpar.get("ppO2_cap_pul_0", 10.0);
      initvals[52] = respirpar.get("ppCO2_ven_pul_0", 5.0);
      initvals[53] = respirpar.get("ppO2_ven_pul_0", 10.0);
      initvals[54] = respirpar.get("ppCO2_at_l_0", 5.0);
      initvals[55] = respirpar.get("ppO2_at_l_0", 10.0);
      initvals[56] = respirpar.get("ppCO2_v_l_0", 5.0);
      initvals[57] = respirpar.get("ppO2_v_l_0", 10.0);
      initvals[58] = respirpar.get("ppCO2_ar_sys_0", 5.0);
      initvals[59] = respirpar.get("ppO2_ar_sys_0", 10.0);
      initvals[60] = respirpar.get("ppCO2_arspl_sys_0", 5.0);
      initvals[61] = respirpar.get("ppO2_arspl_sys_0", 10.0);
      initvals[62] = respirpar.get("ppCO2_arespl_sys_0", 5.0);
      initvals[63] = respirpar.get("ppO2_arespl_sys_0", 10.0);
      initvals[64] = respirpar.get("ppCO2_armsc_sys_0", 5.0);
      initvals[65] = respirpar.get("ppO2_armsc_sys_0", 10.0);
      initvals[66] = respirpar.get("ppCO2_arcer_sys_0", 5.0);
      initvals[67] = respirpar.get("ppO2_arcer_sys_0", 10.0);
      initvals[68] = respirpar.get("ppCO2_arcor_sys_0", 5.0);
      initvals[69] = respirpar.get("ppO2_arcor_sys_0", 10.0);
      initvals[70] = respirpar.get("ppCO2_venspl_sys_0", 5.0);
      initvals[71] = respirpar.get("ppO2_venspl_sys_0", 10.0);
      initvals[72] = respirpar.get("ppCO2_venespl_sys_0", 5.0);
      initvals[73] = respirpar.get("ppO2_venespl_sys_0", 10.0);
      initvals[74] = respirpar.get("ppCO2_venmsc_sys_0", 5.0);
      initvals[75] = respirpar.get("ppO2_venmsc_sys_0", 10.0);
      initvals[76] = respirpar.get("ppCO2_vencer_sys_0", 5.0);
      initvals[77] = respirpar.get("ppO2_vencer_sys_0", 10.0);
      initvals[78] = respirpar.get("ppCO2_vencor_sys_0", 5.0);
      initvals[79] = respirpar.get("ppO2_vencor_sys_0", 10.0);
      initvals[80] = respirpar.get("ppCO2_ven_sys_0", 5.0);
      initvals[81] = respirpar.get("ppO2_ven_sys_0", 10.0);

      break;
  }



  for (int j = 0; j < num_dof_; j++)
  {
    sysvec2->sum_into_global_values(1, &initvals[j], &gindex[j]);
  }

  //----------------------------------------------------------------------
  // loop through conditions and evaluate them if they match the criterion
  //----------------------------------------------------------------------
  for (auto* cond : cardiovascular0dcond_)
  {
    // Get ConditionID of current condition if defined and write value in parameterlist
    int condID = cond->parameters().get<int>("id");
    params.set("id", condID);

    params.set<const Core::Conditions::Condition*>("condition", cond);

    // define element matrices and vectors
    Core::LinAlg::SerialDenseMatrix elematrix1;
    Core::LinAlg::SerialDenseMatrix elematrix2;
    Core::LinAlg::SerialDenseVector elevector1;
    Core::LinAlg::SerialDenseVector elevector2;
    Core::LinAlg::SerialDenseVector elevector3;

    const std::string conditiontype = cond->parameters().get<std::string>("TYPE");

    const std::map<int, std::shared_ptr<Core::Elements::Element>>& geom = cond->geometry();
    // no check for empty geometry here since in parallel computations
    // can exist processors which do not own a portion of the elements belonging
    // to the condition geometry
    for (const auto& [id, ele] : geom)
    {
      // get element location vector and ownerships
      std::vector<int> lm;
      std::vector<int> lmowner;
      std::vector<int> lmstride;
      ele->location_vector(*actdisc_, lm, lmowner, lmstride);

      // get dimension of element matrices and vectors
      // Reshape element matrices and vectors and init to zero
      elevector3.size(1);

      // call the element specific evaluate method
      int err = ele->evaluate(
          params, *actdisc_, lm, elematrix1, elematrix2, elevector1, elevector2, elevector3);
      if (err) FOUR_C_THROW("error while evaluating elements");

      // assembly

      std::vector<int> cardiovascular0dlm;
      std::vector<int> cardiovascular0downer;

      if (conditiontype == "ventricle_left") cardiovascular0dlm.push_back(gindex[2]);
      if (conditiontype == "ventricle_right") cardiovascular0dlm.push_back(gindex[26]);
      if (conditiontype == "atrium_left") cardiovascular0dlm.push_back(gindex[0]);
      if (conditiontype == "atrium_right") cardiovascular0dlm.push_back(gindex[24]);
      cardiovascular0downer.push_back(ele->owner());
      if (assvec1 and conditiontype != "dummy")
        Core::LinAlg::assemble(*sysvec1, elevector3, cardiovascular0dlm, cardiovascular0downer);
    }
  }

  if (Core::Communication::my_mpi_rank(actdisc_->get_comm()) == 0)
  {
    switch (respiratory_model_)
    {
      case Inpar::Cardiovascular0D::resp_none:
      {
        std::cout << "============ Welcome to monolithic coupling of 3D structural dynamics to 0D "
                     "cardiovascular flow models ======================="
                  << std::endl;
        std::cout << "======= Model: Extended closed-loop vascular model with atria (3D or 0D), "
                     "systemic and pulmonary circulation coupling, ======="
                  << std::endl;
        std::cout << "====== including the periphery, each with arterial and venous windkessel "
                     "models; as well as piecewise-linear valve laws ======\n"
                  << std::endl;
      }
      break;
      case Inpar::Cardiovascular0D::resp_standard:
      {
        std::cout << "============ Welcome to monolithic coupling of 3D structural dynamics to 0D "
                     "cardiovascular flow models ======================="
                  << std::endl;
        std::cout << "======= Model: Extended closed-loop vascular model with atria (3D or 0D), "
                     "systemic and pulmonary circulation coupling, ======="
                  << std::endl;
        std::cout << "====== including the periphery, each with arterial and venous windkessel "
                     "models; as well as piecewise-linear valve laws ======"
                  << std::endl;
        std::cout << "======================== PLUS: respiratory model for oxygen and carbon "
                     "dioxide transport and solution ========================\n"
                  << std::endl;
      }
      break;
    }
  }
  return;
}  // end of Initialize Cardiovascular0D

FOUR_C_NAMESPACE_CLOSE
