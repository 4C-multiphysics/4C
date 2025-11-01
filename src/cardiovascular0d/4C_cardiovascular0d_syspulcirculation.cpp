// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_cardiovascular0d_syspulcirculation.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_utils_boundary_integration.hpp"
#include "4C_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_fem_general_utils_nurbs_shapefunctions.hpp"
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
Utils::Cardiovascular0DSysPulCirculation::Cardiovascular0DSysPulCirculation(
    std::shared_ptr<Core::FE::Discretization> discr, const std::string& conditionname,
    std::vector<int>& curID)
    : Cardiovascular0D(discr, conditionname, curID)
{
  Teuchos::ParameterList artvensyspulpar =
      Global::Problem::instance()->cardiovascular0_d_structural_params().sublist(
          "SYS-PUL CIRCULATION PARAMETERS");

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
  c_ar_pul_ = artvensyspulpar.get("C_ar_pul", 0.0);
  r_ar_pul_ = artvensyspulpar.get("R_ar_pul", 0.0);
  l_ar_pul_ = artvensyspulpar.get("L_ar_pul", 0.0);
  z_ar_pul_ = artvensyspulpar.get("Z_ar_pul", 0.0);
  c_ven_sys_ = artvensyspulpar.get("C_ven_sys", 0.0);
  r_ven_sys_ = artvensyspulpar.get("R_ven_sys", 0.0);
  l_ven_sys_ = artvensyspulpar.get("L_ven_sys", 0.0);
  c_ven_pul_ = artvensyspulpar.get("C_ven_pul", 0.0);
  r_ven_pul_ = artvensyspulpar.get("R_ven_pul", 0.0);
  l_ven_pul_ = artvensyspulpar.get("L_ven_pul", 0.0);

  v_v_l_u_ = artvensyspulpar.get("V_v_l_u", 0.0);
  v_at_l_u_ = artvensyspulpar.get("V_at_l_u", 0.0);
  v_ar_sys_u_ = artvensyspulpar.get("V_ar_sys_u", 0.0);
  v_ven_sys_u_ = artvensyspulpar.get("V_ven_sys_u", 0.0);
  v_v_r_u_ = artvensyspulpar.get("V_v_r_u", 0.0);
  v_at_r_u_ = artvensyspulpar.get("V_at_r_u", 0.0);
  v_ar_pul_u_ = artvensyspulpar.get("V_ar_pul_u", 0.0);
  v_ven_pul_u_ = artvensyspulpar.get("V_ven_pul_u", 0.0);
}



/*-----------------------------------------------------------------------*
 |(private)                                                    mhv 02/15 |
 |Evaluate method for a closed-loop 0D vascular model                    |
 |(Hirschvogel, Bassilious, Jagschies, Wildhirt, Gee, "A monolithic 3D-0D|
 |coupled closed-loop model of the heart and the vascular system:        |
 |Experiment-based parameter estimation for patient-specific cardiac     |
 |mechanics", IJNMBE, 2016)                                              |
 *-----------------------------------------------------------------------*/
void Utils::Cardiovascular0DSysPulCirculation::evaluate(Teuchos::ParameterList& params,
    std::shared_ptr<Core::LinAlg::SparseMatrix> sysmat1,
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
  double theta = params.get("scale_theta", 1.0);
  double ts_size = params.get("time_step_size", 1.0);

  // global and local ID of this bc in the redundant vectors
  const int offsetID = params.get<int>("OffsetID");
  std::vector<int> gindex(16);
  gindex[0] = offsetID;
  for (int j = 1; j < 16; j++) gindex[j] = gindex[0] + j;

  bool usetime = true;
  const double tim = params.get("total time", -1.0);
  if (tim < 0.0) usetime = false;

  std::vector<bool> havegid(16);
  for (int j = 0; j < 16; j++)
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
  Core::LinAlg::SerialDenseMatrix wkstiff(16, 16);

  // contributions to total residuals r:
  // r_m = df_m              - f_m
  //     = (df_np - df_n)/dt - theta f_np - (1-theta) f_n
  // here we ONLY evaluate df_np, f_np
  std::vector<double> df_np(16);
  std::vector<double> f_np(16);

  // end-point values at t_{n+1}
  double q_vin_l_np = 0.;
  double p_at_l_np = 0.;
  double q_vout_l_np = 0.;
  double p_v_l_np = 0.;
  double p_ar_sys_np = 0.;
  double q_ar_sys_np = 0.;
  double p_ven_sys_np = 0.;
  double q_ven_sys_np = 0.;
  double q_vin_r_np = 0.;
  double p_at_r_np = 0.;
  double q_vout_r_np = 0.;
  double p_v_r_np = 0.;
  double p_ar_pul_np = 0.;
  double q_ar_pul_np = 0.;
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

  if (assvec1 or assvec2 or assvec4 or assvec5)
  {
    // extract values of dof vector at t_{n+1}
    p_at_l_np = sysvec4->local_values_as_span()[0];
    q_vin_l_np = sysvec4->local_values_as_span()[1];
    q_vout_l_np = sysvec4->local_values_as_span()[2];
    p_v_l_np = sysvec4->local_values_as_span()[3];
    p_ar_sys_np = sysvec4->local_values_as_span()[4];
    q_ar_sys_np = sysvec4->local_values_as_span()[5];
    p_ven_sys_np = sysvec4->local_values_as_span()[6];
    q_ven_sys_np = sysvec4->local_values_as_span()[7];
    p_at_r_np = sysvec4->local_values_as_span()[8];
    q_vin_r_np = sysvec4->local_values_as_span()[9];
    q_vout_r_np = sysvec4->local_values_as_span()[10];
    p_v_r_np = sysvec4->local_values_as_span()[11];
    p_ar_pul_np = sysvec4->local_values_as_span()[12];
    q_ar_pul_np = sysvec4->local_values_as_span()[13];
    p_ven_pul_np = sysvec4->local_values_as_span()[14];
    q_ven_pul_np = sysvec4->local_values_as_span()[15];

    // 3D ventricular volume at t_{n+1}
    V_v_l_np = (*sysvec5).local_values_as_span()[2];
    V_v_r_np = (*sysvec5).local_values_as_span()[10];
    // 3D atrial volume at t_{n+1}
    V_at_l_np = (*sysvec5).local_values_as_span()[0];
    V_at_r_np = (*sysvec5).local_values_as_span()[8];

    switch (atrium_model_)
    {
      case Inpar::Cardiovascular0D::atr_elastance_0d:
      case Inpar::Cardiovascular0D::atr_prescribed:
      {
        df_np[0] = p_at_l_np / E_at_l_np;
        df_np[8] = p_at_r_np / E_at_r_np;
      }
      break;
      case Inpar::Cardiovascular0D::atr_structure_3d:
      {
        df_np[0] = V_at_l_np;
        df_np[8] = V_at_r_np;
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
        df_np[10] = V_v_r_np;
      }
      break;
      case Inpar::Cardiovascular0D::ventr_elastance_0d:
      case Inpar::Cardiovascular0D::ventr_prescribed:
      {
        df_np[2] = p_v_l_np / E_v_l_np;
        df_np[10] = p_v_r_np / E_v_r_np;
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

    df_np[1] = 0.;
    df_np[3] = 0.;
    df_np[4] = c_ar_sys_ * (p_ar_sys_np - z_ar_sys_ * q_vout_l_np);
    df_np[5] = (l_ar_sys_ / r_ar_sys_) * q_ar_sys_np;
    df_np[6] = c_ven_sys_ * p_ven_sys_np;
    df_np[7] = (l_ven_sys_ / r_ven_sys_) * q_ven_sys_np;

    df_np[9] = 0.;
    df_np[11] = 0.;
    df_np[12] = c_ar_pul_ * (p_ar_pul_np - z_ar_pul_ * q_vout_r_np);
    df_np[13] = (l_ar_pul_ / r_ar_pul_) * q_ar_pul_np;
    df_np[14] = c_ven_pul_ * p_ven_pul_np;
    df_np[15] = (l_ven_pul_ / r_ven_pul_) * q_ven_pul_np;

    f_np[0] = -q_ven_pul_np + q_vin_l_np;
    // atrioventricular valve - mitral
    f_np[1] = (p_at_l_np - p_v_l_np) / R_atvalve_l - q_vin_l_np;
    f_np[2] = -q_vin_l_np + q_vout_l_np;
    // semilunar valve - aortic
    f_np[3] = (p_v_l_np - p_ar_sys_np) / R_arvalve_l - q_vout_l_np;
    f_np[4] = -q_vout_l_np + q_ar_sys_np;
    f_np[5] = (p_ven_sys_np - p_ar_sys_np + z_ar_sys_ * q_vout_l_np) / r_ar_sys_ + q_ar_sys_np;
    f_np[6] = -q_ar_sys_np + q_ven_sys_np;
    f_np[7] = (p_at_r_np - p_ven_sys_np) / r_ven_sys_ + q_ven_sys_np;

    f_np[8] = -q_ven_sys_np + q_vin_r_np;
    // atrioventricular valve - tricuspid
    f_np[9] = (p_at_r_np - p_v_r_np) / R_atvalve_r - q_vin_r_np;
    f_np[10] = -q_vin_r_np + q_vout_r_np;
    // semilunar valve - pulmonary
    f_np[11] = (p_v_r_np - p_ar_pul_np) / R_arvalve_r - q_vout_r_np;
    f_np[12] = -q_vout_r_np + q_ar_pul_np;
    f_np[13] = (p_ven_pul_np - p_ar_pul_np + z_ar_pul_ * q_vout_r_np) / r_ar_pul_ + q_ar_pul_np;
    f_np[14] = -q_ar_pul_np + q_ven_pul_np;
    f_np[15] = (p_at_l_np - p_ven_pul_np) / r_ven_pul_ + q_ven_pul_np;
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
        wkstiff(8, 8) = 1. / (E_at_r_np * ts_size);
        break;
      case Inpar::Cardiovascular0D::atr_structure_3d:
        wkstiff(0, 0) = 0.;
        wkstiff(8, 8) = 0.;
        break;
    }

    // ventricle - left and right
    switch (ventricle_model_)
    {
      case Inpar::Cardiovascular0D::ventr_structure_3d:
        wkstiff(2, 3) = 0.;
        wkstiff(10, 11) = 0.;
        break;
      case Inpar::Cardiovascular0D::ventr_elastance_0d:
      case Inpar::Cardiovascular0D::ventr_prescribed:
        wkstiff(2, 3) = 1. / (E_v_l_np * ts_size);
        wkstiff(10, 11) = 1. / (E_v_r_np * ts_size);
        break;
    }

    // atrium - left
    wkstiff(0, 1) = theta;
    wkstiff(0, 15) = -theta;

    // atrioventricular valve - mitral
    wkstiff(1, 1) = -theta;
    wkstiff(1, 0) = theta / R_atvalve_l;
    wkstiff(1, 3) = -theta / R_atvalve_l;

    // ventricular mass balance - left
    wkstiff(2, 2) = theta;
    wkstiff(2, 1) = -theta;

    // semilunar valve - aortic
    wkstiff(3, 3) = theta / R_arvalve_l;
    wkstiff(3, 4) = -theta / R_arvalve_l;
    wkstiff(3, 2) = -theta;

    // arterial mass balance - systemic
    wkstiff(4, 4) = c_ar_sys_ / ts_size;
    wkstiff(4, 2) = -theta - c_ar_sys_ * z_ar_sys_ / ts_size;
    wkstiff(4, 5) = theta;

    // arterial linear momentum balance - systemic
    wkstiff(5, 5) = l_ar_sys_ / (r_ar_sys_ * ts_size) + theta;
    wkstiff(5, 2) = z_ar_sys_ * theta / r_ar_sys_;
    wkstiff(5, 4) = -theta / r_ar_sys_;
    wkstiff(5, 6) = theta / r_ar_sys_;

    // venous mass balance - systemic
    wkstiff(6, 6) = c_ven_sys_ / ts_size;
    wkstiff(6, 5) = -theta;
    wkstiff(6, 7) = theta;

    // venous linear momentum balance - systemic
    wkstiff(7, 7) = l_ven_sys_ / (r_ven_sys_ * ts_size) + theta;
    wkstiff(7, 6) = -theta / r_ven_sys_;
    wkstiff(7, 8) = theta / r_ven_sys_;


    // atrium - right
    wkstiff(8, 9) = theta;
    wkstiff(8, 7) = -theta;

    // atrioventricular valve - tricuspid
    wkstiff(9, 9) = -theta;
    wkstiff(9, 8) = theta / R_atvalve_r;
    wkstiff(9, 11) = -theta / R_atvalve_r;

    // ventricular mass balance - right
    wkstiff(10, 10) = theta;
    wkstiff(10, 9) = -theta;

    // semilunar valve - pulmonary
    wkstiff(11, 11) = theta / R_arvalve_r;
    wkstiff(11, 12) = -theta / R_arvalve_r;
    wkstiff(11, 10) = -theta;

    // arterial mass balance - pulmonary
    wkstiff(12, 12) = c_ar_pul_ / ts_size;
    wkstiff(12, 10) = -theta - c_ar_pul_ * z_ar_pul_ / ts_size;
    wkstiff(12, 13) = theta;

    // arterial linear momentum balance - pulmonary
    wkstiff(13, 13) = l_ar_pul_ / (r_ar_pul_ * ts_size) + theta;
    wkstiff(13, 10) = z_ar_pul_ * theta / r_ar_pul_;
    wkstiff(13, 12) = -theta / r_ar_pul_;
    wkstiff(13, 14) = theta / r_ar_pul_;

    // venous mass balance - pulmonary
    wkstiff(14, 14) = c_ven_pul_ / ts_size;
    wkstiff(14, 13) = -theta;
    wkstiff(14, 15) = theta;

    // venous linear momentum balance - pulmonary
    wkstiff(15, 15) = l_ven_pul_ / (r_ven_pul_ * ts_size) + theta;
    wkstiff(15, 14) = -theta / r_ven_pul_;
    wkstiff(15, 0) = theta / r_ven_pul_;


    sysmat1->un_complete();

    // assemble into cardiovascular0d system matrix - wkstiff contribution
    for (int j = 0; j < 16; j++)
    {
      for (int k = 0; k < 16; k++)
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
    for (int j = 0; j < 16; j++)
    {
      sysvec1->sum_into_global_values(1, &df_np[j], &gindex[j]);
    }
  }
  // rhs part f_np
  if (assvec2)
  {
    for (int j = 0; j < 16; j++)
    {
      sysvec2->sum_into_global_values(1, &f_np[j], &gindex[j]);
    }
  }

  // set vector of compartment volumes - only for post-processing purposes!
  if (assvec4 and assvec5)
  {
    p_at_l_np = sysvec4->local_values_as_span()[0];
    q_vout_l_np = sysvec4->local_values_as_span()[2];
    p_v_l_np = sysvec4->local_values_as_span()[3];
    p_ar_sys_np = sysvec4->local_values_as_span()[4];
    p_ven_sys_np = sysvec4->local_values_as_span()[6];

    p_at_r_np = sysvec4->local_values_as_span()[8];
    q_vout_r_np = sysvec4->local_values_as_span()[10];
    p_v_r_np = sysvec4->local_values_as_span()[11];
    p_ar_pul_np = sysvec4->local_values_as_span()[12];
    p_ven_pul_np = sysvec4->local_values_as_span()[14];

    if (atrium_model_ == Inpar::Cardiovascular0D::atr_elastance_0d or
        atrium_model_ == Inpar::Cardiovascular0D::atr_prescribed)
    {
      // 0D left atrial volume
      (*sysvec5).get_values()[0] = p_at_l_np / E_at_l_np + v_at_l_u_;
      // 0D right atrial volume
      (*sysvec5).get_values()[8] = p_at_r_np / E_at_r_np + v_at_r_u_;
    }
    if (ventricle_model_ == Inpar::Cardiovascular0D::ventr_elastance_0d or
        ventricle_model_ == Inpar::Cardiovascular0D::ventr_prescribed)
    {
      // 0D left ventricular volume
      (*sysvec5).get_values()[2] = p_v_l_np / E_v_l_np + v_v_l_u_;
      // 0D right ventricular volume
      (*sysvec5).get_values()[10] = p_v_r_np / E_v_r_np + v_v_r_u_;
    }
    // systemic arterial compartment volume
    (*sysvec5).get_values()[4] = c_ar_sys_ * (p_ar_sys_np - z_ar_sys_ * q_vout_l_np) + v_ar_sys_u_;
    // systemic venous compartment volume
    (*sysvec5).get_values()[6] = c_ven_sys_ * p_ven_sys_np + v_ven_sys_u_;

    // pulmonary arterial compartment volume
    (*sysvec5).get_values()[12] = c_ar_pul_ * (p_ar_pul_np - z_ar_pul_ * q_vout_r_np) + v_ar_pul_u_;
    // pulmonary venous compartment volume
    (*sysvec5).get_values()[14] = c_ven_pul_ * p_ven_pul_np + v_ven_pul_u_;
  }



  //----------------------------------------------------------------------
  // loop through conditions and evaluate them if they match the criterion
  //----------------------------------------------------------------------
  for (unsigned int i = 0; i < cardiovascular0dcond_.size(); ++i)
  {
    const Core::Conditions::Condition& cond = *(cardiovascular0dcond_[i]);

    // elements might need condition
    params.set<const Core::Conditions::Condition*>("condition", &cond);

    const std::string* conditiontype =
        &cardiovascular0dcond_[i]->parameters().get<std::string>("TYPE");

    // define element matrices and vectors
    Core::LinAlg::SerialDenseMatrix elematrix1;
    Core::LinAlg::SerialDenseMatrix elematrix2;
    Core::LinAlg::SerialDenseVector elevector1;
    Core::LinAlg::SerialDenseVector elevector2;
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
      elevector3.size(1);

      // call the element specific evaluate method
      int err = ele->evaluate(
          params, *actdisc_, lm, elematrix1, elematrix2, elevector1, elevector2, elevector3);
      if (err) FOUR_C_THROW("error while evaluating elements");

      // assembly
      int eid = ele->id();

      if (assmat2 and *conditiontype != "dummy")
      {
        // assemble the offdiagonal stiffness block (1,0 block) arising from dR_cardvasc0d/dd
        // -> this matrix is later on transposed when building the whole block matrix
        std::vector<int> colvec(1);
        if (*conditiontype == "ventricle_left") colvec[0] = gindex[2];
        if (*conditiontype == "ventricle_right") colvec[0] = gindex[10];
        if (*conditiontype == "atrium_left") colvec[0] = gindex[0];
        if (*conditiontype == "atrium_right") colvec[0] = gindex[8];
        elevector2.scale(-1. / ts_size);
        sysmat2->assemble(eid, lmstride, elevector2, lm, lmowner, colvec);
      }

      if (assvec3 and *conditiontype != "dummy")
      {
        // assemble the current volume of the enclosed surface of the cardiovascular0d condition
        std::vector<int> cardiovascular0dlm;
        std::vector<int> cardiovascular0downer;

        if (*conditiontype == "ventricle_left") cardiovascular0dlm.push_back(gindex[2]);
        if (*conditiontype == "ventricle_right") cardiovascular0dlm.push_back(gindex[10]);
        if (*conditiontype == "atrium_left") cardiovascular0dlm.push_back(gindex[0]);
        if (*conditiontype == "atrium_right") cardiovascular0dlm.push_back(gindex[8]);
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

  return;
}  // end of evaluate_condition



/*-----------------------------------------------------------------------*
 *-----------------------------------------------------------------------*/
void Utils::Cardiovascular0DSysPulCirculation::initialize(Teuchos::ParameterList& params,
    std::shared_ptr<Core::LinAlg::Vector<double>> sysvec1,
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
  std::vector<int> gindex(16);
  gindex[0] = offsetID;
  for (int j = 1; j < 16; j++) gindex[j] = gindex[0] + j;


  Teuchos::ParameterList artvensyspulpar =
      Global::Problem::instance()->cardiovascular0_d_structural_params().sublist(
          "SYS-PUL CIRCULATION PARAMETERS");

  const double p_at_l_0 = artvensyspulpar.get("p_at_l_0", 0.0);
  const double q_vin_l_0 = artvensyspulpar.get("q_vin_l_0", 0.0);
  const double q_vout_l_0 = artvensyspulpar.get("q_vout_l_0", 0.0);
  const double p_v_l_0 = artvensyspulpar.get("p_v_l_0", 0.0);
  const double p_ar_sys_0 = artvensyspulpar.get("p_ar_sys_0", 0.0);
  const double q_ar_sys_0 = artvensyspulpar.get("q_ar_sys_0", 0.0);
  const double p_ven_sys_0 = artvensyspulpar.get("p_ven_sys_0", 0.0);
  const double q_ven_sys_0 = artvensyspulpar.get("q_ven_sys_0", 0.0);
  const double p_at_r_0 = artvensyspulpar.get("p_at_r_0", 0.0);
  const double q_vin_r_0 = artvensyspulpar.get("q_vin_r_0", 0.0);
  const double q_vout_r_0 = artvensyspulpar.get("q_vout_r_0", 0.0);
  const double p_v_r_0 = artvensyspulpar.get("p_v_r_0", 0.0);
  const double p_ar_pul_0 = artvensyspulpar.get("p_ar_pul_0", 0.0);
  const double q_ar_pul_0 = artvensyspulpar.get("q_ar_pul_0", 0.0);
  const double p_ven_pul_0 = artvensyspulpar.get("p_ven_pul_0", 0.0);
  const double q_ven_pul_0 = artvensyspulpar.get("q_ven_pul_0", 0.0);

  sysvec2->sum_into_global_values(1, &p_at_l_0, &gindex[0]);
  sysvec2->sum_into_global_values(1, &q_vin_l_0, &gindex[1]);
  sysvec2->sum_into_global_values(1, &q_vout_l_0, &gindex[2]);
  sysvec2->sum_into_global_values(1, &p_v_l_0, &gindex[3]);
  sysvec2->sum_into_global_values(1, &p_ar_sys_0, &gindex[4]);
  sysvec2->sum_into_global_values(1, &q_ar_sys_0, &gindex[5]);
  sysvec2->sum_into_global_values(1, &p_ven_sys_0, &gindex[6]);
  sysvec2->sum_into_global_values(1, &q_ven_sys_0, &gindex[7]);
  sysvec2->sum_into_global_values(1, &p_at_r_0, &gindex[8]);
  sysvec2->sum_into_global_values(1, &q_vin_r_0, &gindex[9]);
  sysvec2->sum_into_global_values(1, &q_vout_r_0, &gindex[10]);
  sysvec2->sum_into_global_values(1, &p_v_r_0, &gindex[11]);
  sysvec2->sum_into_global_values(1, &p_ar_pul_0, &gindex[12]);
  sysvec2->sum_into_global_values(1, &q_ar_pul_0, &gindex[13]);
  sysvec2->sum_into_global_values(1, &p_ven_pul_0, &gindex[14]);
  sysvec2->sum_into_global_values(1, &q_ven_pul_0, &gindex[15]);

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

    const auto& geom = cond->geometry();
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
      if (conditiontype == "ventricle_right") cardiovascular0dlm.push_back(gindex[10]);
      if (conditiontype == "atrium_left") cardiovascular0dlm.push_back(gindex[0]);
      if (conditiontype == "atrium_right") cardiovascular0dlm.push_back(gindex[8]);
      cardiovascular0downer.push_back(ele->owner());
      if (assvec1 and conditiontype != "dummy")
        Core::LinAlg::assemble(*sysvec1, elevector3, cardiovascular0dlm, cardiovascular0downer);
    }
  }

  if (Core::Communication::my_mpi_rank(actdisc_->get_comm()) == 0)
  {
    std::cout << "============ Welcome to monolithic coupling of 3D structural dynamics to 0D "
                 "cardiovascular flow models ============"
              << std::endl;
    std::cout << "====== Model: Closed-loop vascular model with atria (3D or 0D), systemic and "
                 "pulmonary circulation coupling, ======"
              << std::endl;
    std::cout << "=============== each with arterial and venous windkessel models; as well as "
                 "piecewise-linear valve laws ===========\n"
              << std::endl;
  }
  return;
}  // end of Initialize Cardiovascular0D

FOUR_C_NAMESPACE_CLOSE
