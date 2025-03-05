// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_inpar_cardiovascular0d.hpp"

#include "4C_fem_condition_definition.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN



void Inpar::Cardiovascular0D::set_valid_parameters(std::map<std::string, Core::IO::InputSpec>& list)
{
  using Teuchos::tuple;
  using namespace Core::IO::InputSpecBuilders;

  Core::Utils::SectionSpecs cardvasc0dstruct{"CARDIOVASCULAR 0D-STRUCTURE COUPLING"};

  cardvasc0dstruct.specs.emplace_back(parameter<double>("TOL_CARDVASC0D_RES",
      {.description = "tolerance in the cardiovascular0d error norm for the Newton iteration",
          .default_value = 1.0E-08}));
  cardvasc0dstruct.specs.emplace_back(parameter<double>("TOL_CARDVASC0D_DOFINCR",
      {.description =
              "tolerance in the cardiovascular0d dof increment error norm for the Newton iteration",
          .default_value = 1.0E-08}));
  cardvasc0dstruct.specs.emplace_back(parameter<double>("TIMINT_THETA",
      {.description = "theta for one-step-theta time-integration scheme of Cardiovascular0D",
          .default_value = 0.5}));
  cardvasc0dstruct.specs.emplace_back(parameter<bool>("RESTART_WITH_CARDVASC0D",
      {.description = "Must be chosen if a non-cardiovascular0d simulation is to be restarted as "
                      "cardiovascular0d-structural coupled problem.",
          .default_value = false}));
  cardvasc0dstruct.specs.emplace_back(parameter<bool>("ENHANCED_OUTPUT",
      {.description = "Set to yes for enhanced output (like e.g. derivative information)",
          .default_value = false}));

  // linear solver id used for monolithic 0D cardiovascular-structural problems
  Core::Utils::int_parameter("LINEAR_COUPLED_SOLVER", -1,
      "number of linear solver used for cardiovascular 0D-structural problems", cardvasc0dstruct);

  Core::Utils::string_to_integral_parameter<Cardvasc0DSolveAlgo>("SOLALGORITHM", "direct", "",
      tuple<std::string>("block", "direct"),
      tuple<Cardvasc0DSolveAlgo>(Inpar::Cardiovascular0D::cardvasc0dsolve_block,
          Inpar::Cardiovascular0D::cardvasc0dsolve_direct),
      cardvasc0dstruct);

  cardvasc0dstruct.specs.emplace_back(
      parameter<double>("T_PERIOD", {.description = "periodic time", .default_value = -1.0}));
  cardvasc0dstruct.specs.emplace_back(parameter<double>(
      "EPS_PERIODIC", {.description = "tolerance for periodic state", .default_value = 1.0e-16}));

  cardvasc0dstruct.specs.emplace_back(parameter<bool>("PTC_3D0D",
      {.description = "Set to yes for doing PTC 2x2 block system.", .default_value = false}));

  cardvasc0dstruct.specs.emplace_back(parameter<double>(
      "K_PTC", {.description = "PTC parameter: 0 means normal Newton, ->infty means steepest desc",
                   .default_value = 0.0}));

  cardvasc0dstruct.move_into_collection(list);

  Core::Utils::SectionSpecs cardvasc0dsyspulcirc{
      cardvasc0dstruct, "SYS-PUL CIRCULATION PARAMETERS"};

  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>("R_arvalve_max_l",
      {.description = "maximal left arterial (semilunar) valve resistance", .default_value = 0.0}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>("R_arvalve_min_l",
      {.description = "minimal left arterial (semilunar) valve resistance", .default_value = 0.0}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>(
      "R_atvalve_max_l", {.description = "maximal left atrial (atrioventricular) valve resistance",
                             .default_value = 0.0}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>(
      "R_atvalve_min_l", {.description = "minimal left atrial (atrioventricular) valve resistance",
                             .default_value = 0.0}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>(
      "R_arvalve_max_r", {.description = "maximal right arterial (semilunar) valve resistance",
                             .default_value = 0.0}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>(
      "R_arvalve_min_r", {.description = "minimal right arterial (semilunar) valve resistance",
                             .default_value = 0.0}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>(
      "R_atvalve_max_r", {.description = "maximal right atrial (atrioventricular) valve resistance",
                             .default_value = 0.0}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>(
      "R_atvalve_min_r", {.description = "minimal right atrial (atrioventricular) valve resistance",
                             .default_value = 0.0}));

  Core::Utils::string_to_integral_parameter<Cardvasc0DAtriumModel>("ATRIUM_MODEL", "0D", "",
      tuple<std::string>("0D", "3D", "prescribed"),
      tuple<Cardvasc0DAtriumModel>(Inpar::Cardiovascular0D::atr_elastance_0d,
          Inpar::Cardiovascular0D::atr_structure_3d, Inpar::Cardiovascular0D::atr_prescribed),
      cardvasc0dsyspulcirc);
  Core::Utils::int_parameter("Atrium_act_curve_l", -1,
      "left atrial activation curve (ONLY for ATRIUM_MODEL '0D'!)", cardvasc0dsyspulcirc);
  Core::Utils::int_parameter("Atrium_act_curve_r", -1,
      "right atrial activation curve (ONLY for ATRIUM_MODEL '0D'!)", cardvasc0dsyspulcirc);
  Core::Utils::int_parameter("Atrium_prescr_E_curve_l", -1,
      "left atrial elastance prescription curve (ONLY for ATRIUM_MODEL 'prescribed'!)",
      cardvasc0dsyspulcirc);
  Core::Utils::int_parameter("Atrium_prescr_E_curve_r", -1,
      "right atrial elastance prescription curve (ONLY for ATRIUM_MODEL 'prescribed'!)",
      cardvasc0dsyspulcirc);
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>(
      "E_at_max_l", {.description = "0D maximum left atrial elastance", .default_value = 0.0}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>(
      "E_at_min_l", {.description = "0D baseline left atrial elastance", .default_value = 0.0}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>(
      "E_at_max_r", {.description = "0D maximum right atrial elastance", .default_value = 0.0}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>(
      "E_at_min_r", {.description = "0D baseline right atrial elastance", .default_value = 0.0}));

  Core::Utils::string_to_integral_parameter<Cardvasc0DVentricleModel>("VENTRICLE_MODEL", "3D", "",
      tuple<std::string>("3D", "0D", "prescribed"),
      tuple<Cardvasc0DVentricleModel>(Inpar::Cardiovascular0D::ventr_structure_3d,
          Inpar::Cardiovascular0D::ventr_elastance_0d, Inpar::Cardiovascular0D::ventr_prescribed),
      cardvasc0dsyspulcirc);
  Core::Utils::int_parameter("Ventricle_act_curve_l", -1,
      "left ventricular activation curve (ONLY for VENTRICLE_MODEL '0D'!)", cardvasc0dsyspulcirc);
  Core::Utils::int_parameter("Ventricle_act_curve_r", -1,
      "right ventricular activation curve (ONLY for VENTRICLE_MODEL '0D'!)", cardvasc0dsyspulcirc);
  Core::Utils::int_parameter("Ventricle_prescr_E_curve_l", -1,
      "left ventricular elastance prescription curve (ONLY for VENTRICLE_MODEL 'prescribed'!)",
      cardvasc0dsyspulcirc);
  Core::Utils::int_parameter("Ventricle_prescr_E_curve_r", -1,
      "right ventricular elastance prescription curve (ONLY for VENTRICLE_MODEL 'prescribed'!)",
      cardvasc0dsyspulcirc);
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>(
      "E_v_max_l", {.description = "0D maximum left ventricular elastance", .default_value = 0.0}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>("E_v_min_l",
      {.description = "0D baseline left ventricular elastance", .default_value = 0.0}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>("E_v_max_r",
      {.description = "0D maximum right ventricular elastance", .default_value = 0.0}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>("E_v_min_r",
      {.description = "0D baseline right ventricular elastance", .default_value = 0.0}));

  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>(
      "C_ar_sys", {.description = "systemic arterial compliance", .default_value = 0.0}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>(
      "R_ar_sys", {.description = "systemic arterial resistance", .default_value = 0.0}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>(
      "L_ar_sys", {.description = "systemic arterial inertance", .default_value = 0.0}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>(
      "Z_ar_sys", {.description = "systemic arterial impedance", .default_value = 0.0}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>(
      "C_ar_pul", {.description = "pulmonary arterial compliance", .default_value = 0.0}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>(
      "R_ar_pul", {.description = "pulmonary arterial resistance", .default_value = 0.0}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>(
      "L_ar_pul", {.description = "pulmonary arterial inertance", .default_value = 0.0}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>(
      "Z_ar_pul", {.description = "pulmonary arterial impedance", .default_value = 0.0}));

  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>(
      "C_ven_sys", {.description = "systemic venous compliance", .default_value = 0.0}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>(
      "R_ven_sys", {.description = "systemic venous resistance", .default_value = 0.0}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>(
      "L_ven_sys", {.description = "systemic venous inertance", .default_value = 0.0}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>(
      "C_ven_pul", {.description = "pulmonary venous compliance", .default_value = 0.0}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>(
      "R_ven_pul", {.description = "pulmonary venous resistance", .default_value = 0.0}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>(
      "L_ven_pul", {.description = "pulmonary venous inertance", .default_value = 0.0}));

  // initial conditions
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>(
      "q_vin_l_0", {.description = "initial left ventricular in-flux", .default_value = 0.0}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>(
      "p_at_l_0", {.description = "initial left atrial pressure", .default_value = 0.0}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>(
      "q_vout_l_0", {.description = "initial left ventricular out-flux", .default_value = 0.0}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>(
      "p_v_l_0", {.description = "initial left ventricular pressure", .default_value = 0.0}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>(
      "p_ar_sys_0", {.description = "initial systemic arterial pressure", .default_value = 0.0}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>(
      "q_ar_sys_0", {.description = "initial systemic arterial flux", .default_value = 0.0}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>(
      "p_ven_sys_0", {.description = "initial systemic venous pressure", .default_value = 0.0}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>(
      "q_ven_sys_0", {.description = "initial systemic venous flux", .default_value = 0.0}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>(
      "q_vin_r_0", {.description = "initial right ventricular in-flux", .default_value = 0.0}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>(
      "p_at_r_0", {.description = "initial right atrial pressure", .default_value = 0.0}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>(
      "q_vout_r_0", {.description = "initial right ventricular out-flux", .default_value = 0.0}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>(
      "p_v_r_0", {.description = "initial right ventricular pressure", .default_value = 0.0}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>(
      "p_ar_pul_0", {.description = "initial pulmonary arterial pressure", .default_value = 0.0}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>(
      "q_ar_pul_0", {.description = "initial pulmonary arterial flux", .default_value = 0.0}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>(
      "p_ven_pul_0", {.description = "initial pulmonary venous pressure", .default_value = 0.0}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>(
      "q_ven_pul_0", {.description = "initial pulmonary venous flux", .default_value = 0.0}));

  // unstressed volumes - only for postprocessing matters!
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>(
      "V_at_l_u", {.description = "unstressed volume of left 0D atrium", .default_value = 0.0}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>(
      "V_v_l_u", {.description = "unstressed volume of left 0D ventricle", .default_value = 0.0}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>(
      "V_ar_sys_u", {.description = "unstressed volume of systemic arteries and capillaries",
                        .default_value = 0.0}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>("V_ven_sys_u",
      {.description = "unstressed volume of systemic veins", .default_value = 100.0e3}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>(
      "V_at_r_u", {.description = "unstressed volume of right 0D atrium", .default_value = 0.0}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>(
      "V_v_r_u", {.description = "unstressed volume of right 0D ventricle", .default_value = 0.0}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>(
      "V_ar_pul_u", {.description = "unstressed volume of pulmonary arteries and capillaries",
                        .default_value = 0.0}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>("V_ven_pul_u",
      {.description = "unstressed volume of pulmonary veins", .default_value = 120.0e3}));



  // parameters for extended sys pul circulation including periphery
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>("C_arspl_sys",
      {.description = "systemic arterial splanchnic compliance", .default_value = 0.0}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>("R_arspl_sys",
      {.description = "systemic arterial splanchnic resistance", .default_value = 0.0}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>("C_arespl_sys",
      {.description = "systemic arterial extra-splanchnic compliance", .default_value = 0.0}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>("R_arespl_sys",
      {.description = "systemic arterial extra-splanchnic resistance", .default_value = 0.0}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>("C_armsc_sys",
      {.description = "systemic arterial muscular compliance", .default_value = 0.0}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>("R_armsc_sys",
      {.description = "systemic arterial muscular resistance", .default_value = 0.0}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>("C_arcer_sys",
      {.description = "systemic arterial cerebral compliance", .default_value = 0.0}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>("R_arcer_sys",
      {.description = "systemic arterial cerebral resistance", .default_value = 0.0}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>("C_arcor_sys",
      {.description = "systemic arterial coronary compliance", .default_value = 0.0}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>("R_arcor_sys",
      {.description = "systemic arterial coronary resistance", .default_value = 0.0}));

  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>("C_venspl_sys",
      {.description = "systemic venous splanchnic compliance", .default_value = 0.0}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>("R_venspl_sys",
      {.description = "systemic venous splanchnic resistance", .default_value = 0.0}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>("C_venespl_sys",
      {.description = "systemic venous extra-splanchnic compliance", .default_value = 0.0}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>("R_venespl_sys",
      {.description = "systemic venous extra-splanchnic resistance", .default_value = 0.0}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>("C_venmsc_sys",
      {.description = "systemic venous muscular compliance", .default_value = 0.0}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>("R_venmsc_sys",
      {.description = "systemic venous muscular resistance", .default_value = 0.0}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>("C_vencer_sys",
      {.description = "systemic venous cerebral compliance", .default_value = 0.0}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>("R_vencer_sys",
      {.description = "systemic venous cerebral resistance", .default_value = 0.0}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>("C_vencor_sys",
      {.description = "systemic venous coronary compliance", .default_value = 0.0}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>("R_vencor_sys",
      {.description = "systemic venous coronary resistance", .default_value = 0.0}));

  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>(
      "C_cap_pul", {.description = "pulmonary capillary compliance", .default_value = 0.0}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>(
      "R_cap_pul", {.description = "pulmonary capillary resistance", .default_value = 0.0}));

  // initial conditions for extended sys pul circulation including periphery
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>("p_arperi_sys_0",
      {.description = "initial systemic peripheral arterial pressure", .default_value = 0.0}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>("q_arspl_sys_0",
      {.description = "initial systemic arterial splanchnic flux", .default_value = 0.0}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>("q_arespl_sys_0",
      {.description = "initial systemic arterial extra-splanchnic flux", .default_value = 0.0}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>("q_armsc_sys_0",
      {.description = "initial systemic arterial muscular flux", .default_value = 0.0}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>("q_arcer_sys_0",
      {.description = "initial systemic arterial cerebral flux", .default_value = 0.0}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>("q_arcor_sys_0",
      {.description = "initial systemic arterial coronary flux", .default_value = 0.0}));

  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>("p_venspl_sys_0",
      {.description = "initial systemic venous splanchnic pressure", .default_value = 0.0}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>("q_venspl_sys_0",
      {.description = "initial systemic venous splanchnic flux", .default_value = 0.0}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>("p_venespl_sys_0",
      {.description = "initial systemic venous extra-splanchnic pressure", .default_value = 0.0}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>("q_venespl_sys_0",
      {.description = "initial systemic venous extra-splanchnic flux", .default_value = 0.0}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>("p_venmsc_sys_0",
      {.description = "initial systemic venous muscular pressure", .default_value = 0.0}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>("q_venmsc_sys_0",
      {.description = "initial systemic venous muscular flux", .default_value = 0.0}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>("p_vencer_sys_0",
      {.description = "initial systemic venous cerebral pressure", .default_value = 0.0}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>("q_vencer_sys_0",
      {.description = "initial systemic venous cerebral flux", .default_value = 0.0}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>("p_vencor_sys_0",
      {.description = "initial systemic venous coronary pressure", .default_value = 0.0}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>("q_vencor_sys_0",
      {.description = "initial systemic venous coronary flux", .default_value = 0.0}));

  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>("p_cap_pul_0",
      {.description = "initial pulmonary capillary pressure", .default_value = 0.0}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>(
      "q_cap_pul_0", {.description = "initial pulmonary capillary flux", .default_value = 0.0}));


  // unstressed volumes
  // default values according to Ursino et al. Am J Physiol Heart Circ Physiol (2000), in mm^3
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>(
      "V_arspl_sys_u", {.description = "unstressed volume of systemic splanchnic arteries",
                           .default_value = 274.4e3}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>(
      "V_arespl_sys_u", {.description = "unstressed volume of systemic extra-splanchnic arteries",
                            .default_value = 134.64e3}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>(
      "V_armsc_sys_u", {.description = "unstressed volume of systemic muscular arteries",
                           .default_value = 105.8e3}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>(
      "V_arcer_sys_u", {.description = "unstressed volume of systemic cerebral arteries",
                           .default_value = 72.13e3}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>("V_arcor_sys_u",
      {.description = "unstressed volume of systemic coronary arteries", .default_value = 24.0e3}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>(
      "V_venspl_sys_u", {.description = "unstressed volume of systemic splanchnic veins",
                            .default_value = 1121.0e3}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>(
      "V_venespl_sys_u", {.description = "unstressed volume of systemic extra-splanchnic veins",
                             .default_value = 550.0e3}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>("V_venmsc_sys_u",
      {.description = "unstressed volume of systemic muscular veins", .default_value = 432.14e3}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>("V_vencer_sys_u",
      {.description = "unstressed volume of systemic cerebral veins", .default_value = 294.64e3}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>("V_vencor_sys_u",
      {.description = "unstressed volume of systemic coronary veins", .default_value = 98.21e3}));
  cardvasc0dsyspulcirc.specs.emplace_back(parameter<double>("V_cap_pul_u",
      {.description = "unstressed volume of pulmonary capillaries", .default_value = 123.0e3}));

  cardvasc0dsyspulcirc.move_into_collection(list);



  Core::Utils::SectionSpecs cardvascrespir0d{cardvasc0dstruct, "RESPIRATORY PARAMETERS"};

  Core::Utils::string_to_integral_parameter<Cardvasc0DRespiratoryModel>("RESPIRATORY_MODEL", "None",
      "", tuple<std::string>("None", "Standard"),
      tuple<Cardvasc0DRespiratoryModel>(
          Inpar::Cardiovascular0D::resp_none, Inpar::Cardiovascular0D::resp_standard),
      cardvascrespir0d);


  cardvascrespir0d.specs.emplace_back(
      parameter<double>("L_alv", {.description = "alveolar inertance", .default_value = 0.0}));
  cardvascrespir0d.specs.emplace_back(
      parameter<double>("R_alv", {.description = "alveolar resistance", .default_value = 0.0}));
  cardvascrespir0d.specs.emplace_back(
      parameter<double>("E_alv", {.description = "alveolar elastance", .default_value = 0.0}));

  cardvascrespir0d.specs.emplace_back(parameter<double>("V_lung_tidal",
      {.description = "tidal volume (the total volume of inspired air, in a single breath)",
          .default_value = 0.4e6}));
  cardvascrespir0d.specs.emplace_back(parameter<double>(
      "V_lung_dead", {.description = "dead space volume", .default_value = 0.15e6}));
  cardvascrespir0d.specs.emplace_back(
      parameter<double>("V_lung_u", {.description = "unstressed lung volume (volume of the lung "
                                                    "when it is fully collapsed outside the body)",
                                        .default_value = 0.0}));

  Core::Utils::int_parameter("U_t_curve", -1,
      "time-varying, prescribed pleural pressure curve driven by diaphragm", cardvascrespir0d);
  cardvascrespir0d.specs.emplace_back(
      parameter<double>("U_m", {.description = "in-breath pressure", .default_value = 0.0}));

  cardvascrespir0d.specs.emplace_back(parameter<double>(
      "fCO2_ext", {.description = "atmospheric CO2 gas fraction", .default_value = 0.03}));
  cardvascrespir0d.specs.emplace_back(parameter<double>(
      "fO2_ext", {.description = "atmospheric O2 gas fraction", .default_value = 0.21}));

  cardvascrespir0d.specs.emplace_back(parameter<double>(
      "kappa_CO2", {.description = "diffusion coefficient for CO2 across the hemato-alveolar "
                                   "membrane, in molar value / (time * pressure)",
                       .default_value = 0.0}));
  cardvascrespir0d.specs.emplace_back(parameter<double>(
      "kappa_O2", {.description = "diffusion coefficient for O2 across the hemato-alveolar "
                                  "membrane, in molar value / (time * pressure)",
                      .default_value = 0.0}));

  // should be 22.4 liters per mol !
  // however we specify it as an input parameter since its decimal power depends on the system of
  // units your whole model is specified in! i.e. if you have kg - mm - s - mmol, it's 22.4e3 mm^3 /
  // mmol
  cardvascrespir0d.specs.emplace_back(parameter<double>(
      "V_m_gas", {.description = "molar volume of an ideal gas", .default_value = 22.4e3}));

  // should be 47.1 mmHg = 6.279485 kPa !
  // however we specify it as an input parameter since its decimal power depends on the system of
  // units your whole model is specified in! i.e. if you have kg - mm - s - mmol, it's 6.279485 kPa
  cardvascrespir0d.specs.emplace_back(parameter<double>(
      "p_vap_water_37", {.description = "vapor pressure of water at 37  degrees celsius",
                            .default_value = 6.279485}));

  cardvascrespir0d.specs.emplace_back(parameter<double>(
      "alpha_CO2", {.description = "CO2 solubility constant, in molar value / (volume * pressure)",
                       .default_value = 0.0}));
  cardvascrespir0d.specs.emplace_back(parameter<double>(
      "alpha_O2", {.description = "O2 solubility constant, in molar value / (volume * pressure)",
                      .default_value = 0.0}));

  cardvascrespir0d.specs.emplace_back(parameter<double>(
      "c_Hb", {.description = "hemoglobin concentration of the blood, in molar value / volume",
                  .default_value = 0.0}));


  cardvascrespir0d.specs.emplace_back(parameter<double>("M_CO2_arspl",
      {.description = "splanchnic metabolic rate of CO2 production", .default_value = 0.0}));
  cardvascrespir0d.specs.emplace_back(parameter<double>("M_O2_arspl",
      {.description = "splanchnic metabolic rate of O2 consumption", .default_value = 0.0}));
  cardvascrespir0d.specs.emplace_back(parameter<double>("M_CO2_arespl",
      {.description = "extra-splanchnic metabolic rate of CO2 production", .default_value = 0.0}));
  cardvascrespir0d.specs.emplace_back(parameter<double>("M_O2_arespl",
      {.description = "extra-splanchnic metabolic rate of O2 consumption", .default_value = 0.0}));
  cardvascrespir0d.specs.emplace_back(parameter<double>("M_CO2_armsc",
      {.description = "muscular metabolic rate of CO2 production", .default_value = 0.0}));
  cardvascrespir0d.specs.emplace_back(parameter<double>("M_O2_armsc",
      {.description = "muscular metabolic rate of O2 consumption", .default_value = 0.0}));
  cardvascrespir0d.specs.emplace_back(parameter<double>("M_CO2_arcer",
      {.description = "cerebral metabolic rate of CO2 production", .default_value = 0.0}));
  cardvascrespir0d.specs.emplace_back(parameter<double>("M_O2_arcer",
      {.description = "cerebral metabolic rate of O2 consumption", .default_value = 0.0}));
  cardvascrespir0d.specs.emplace_back(parameter<double>("M_CO2_arcor",
      {.description = "coronary metabolic rate of CO2 production", .default_value = 0.0}));
  cardvascrespir0d.specs.emplace_back(parameter<double>("M_O2_arcor",
      {.description = "coronary metabolic rate of O2 consumption", .default_value = 0.0}));

  cardvascrespir0d.specs.emplace_back(parameter<double>(
      "V_tissspl", {.description = "splanchnic tissue volume", .default_value = 1.0}));
  cardvascrespir0d.specs.emplace_back(parameter<double>(
      "V_tissespl", {.description = "extra-splanchnic tissue volume", .default_value = 1.0}));
  cardvascrespir0d.specs.emplace_back(parameter<double>(
      "V_tissmsc", {.description = "muscular tissue volume", .default_value = 1.0}));
  cardvascrespir0d.specs.emplace_back(parameter<double>(
      "V_tisscer", {.description = "cerebral tissue volume", .default_value = 1.0}));
  cardvascrespir0d.specs.emplace_back(parameter<double>(
      "V_tisscor", {.description = "coronary tissue volume", .default_value = 1.0}));


  // initial conditions for respiratory model
  cardvascrespir0d.specs.emplace_back(parameter<double>(
      "V_alv_0", {.description = "initial alveolar volume", .default_value = -1.0}));
  cardvascrespir0d.specs.emplace_back(
      parameter<double>("q_alv_0", {.description = "initial alveolar flux", .default_value = 0.0}));
  cardvascrespir0d.specs.emplace_back(parameter<double>(
      "p_alv_0", {.description = "initial alveolar pressure", .default_value = -1.0}));

  cardvascrespir0d.specs.emplace_back(parameter<double>(
      "fCO2_alv_0", {.description = "initial alveolar CO2 fraction", .default_value = 0.05263}));
  cardvascrespir0d.specs.emplace_back(parameter<double>(
      "fO2_alv_0", {.description = "initial alveolar O2 fraction", .default_value = 0.1368}));

  cardvascrespir0d.specs.emplace_back(parameter<double>("q_arspl_sys_in_0",
      {.description = "initial arterial splanchnic in-flux", .default_value = 0.0}));
  cardvascrespir0d.specs.emplace_back(parameter<double>("q_arsspl_sys_in_0",
      {.description = "initial arterial extra-splanchnic in-flux", .default_value = 0.0}));
  cardvascrespir0d.specs.emplace_back(parameter<double>("q_armsc_sys_in_0",
      {.description = "initial arterial muscular in-flux", .default_value = 0.0}));
  cardvascrespir0d.specs.emplace_back(parameter<double>("q_arcer_sys_in_0",
      {.description = "initial arterial cerebral in-flux", .default_value = 0.0}));
  cardvascrespir0d.specs.emplace_back(parameter<double>("q_arcor_sys_in_0",
      {.description = "initial arterial coronary in-flux", .default_value = 0.0}));

  cardvascrespir0d.specs.emplace_back(parameter<double>("ppCO2_at_r_0",
      {.description = "initial right atrial CO2 partial pressure", .default_value = 1.0}));
  cardvascrespir0d.specs.emplace_back(parameter<double>("ppO2_at_r_0",
      {.description = "initial right atrial O2 partial pressure", .default_value = 1.0}));
  cardvascrespir0d.specs.emplace_back(parameter<double>("ppCO2_v_r_0",
      {.description = "initial right ventricular CO2 partial pressure", .default_value = 1.0}));
  cardvascrespir0d.specs.emplace_back(parameter<double>("ppO2_v_r_0",
      {.description = "initial right ventricular O2 partial pressure", .default_value = 1.0}));
  cardvascrespir0d.specs.emplace_back(parameter<double>("ppCO2_ar_pul_0",
      {.description = "initial pulmonary arterial CO2 partial pressure", .default_value = 1.0}));
  cardvascrespir0d.specs.emplace_back(parameter<double>("ppO2_ar_pul_0",
      {.description = "initial pulmonary arterial O2 partial pressure", .default_value = 1.0}));
  cardvascrespir0d.specs.emplace_back(parameter<double>("ppCO2_cap_pul_0",
      {.description = "initial pulmonary capillary CO2 partial pressure", .default_value = 1.0}));
  cardvascrespir0d.specs.emplace_back(parameter<double>("ppO2_cap_pul_0",
      {.description = "initial pulmonary capillary O2 partial pressure", .default_value = 1.0}));
  cardvascrespir0d.specs.emplace_back(parameter<double>("ppCO2_ven_pul_0",
      {.description = "initial pulmonary venous CO2 partial pressure", .default_value = 1.0}));
  cardvascrespir0d.specs.emplace_back(parameter<double>("ppO2_ven_pul_0",
      {.description = "initial pulmonary venous O2 partial pressure", .default_value = 1.0}));
  cardvascrespir0d.specs.emplace_back(parameter<double>("ppCO2_at_l_0",
      {.description = "initial left atrial CO2 partial pressure", .default_value = 1.0}));
  cardvascrespir0d.specs.emplace_back(parameter<double>("ppO2_at_l_0",
      {.description = "initial left atrial O2 partial pressure", .default_value = 1.0}));
  cardvascrespir0d.specs.emplace_back(parameter<double>("ppCO2_v_l_0",
      {.description = "initial left ventricular CO2 partial pressure", .default_value = 1.0}));
  cardvascrespir0d.specs.emplace_back(parameter<double>("ppO2_v_l_0",
      {.description = "initial left ventricular O2 partial pressure", .default_value = 1.0}));
  cardvascrespir0d.specs.emplace_back(parameter<double>("ppCO2_ar_sys_0",
      {.description = "initial systemic arterial CO2 partial pressure", .default_value = 1.0}));
  cardvascrespir0d.specs.emplace_back(parameter<double>("ppO2_ar_sys_0",
      {.description = "initial systemic arterial O2 partial pressure", .default_value = 1.0}));
  cardvascrespir0d.specs.emplace_back(parameter<double>("ppCO2_arspl_sys_0",
      {.description = "initial systemic arterial splanchnic CO2 partial pressure",
          .default_value = 1.0}));
  cardvascrespir0d.specs.emplace_back(parameter<double>("ppO2_arspl_sys_0",
      {.description = "initial systemic arterial splanchnic O2 partial pressure",
          .default_value = 1.0}));
  cardvascrespir0d.specs.emplace_back(parameter<double>("ppCO2_arespl_sys_0",
      {.description = "initial systemic arterial extra-splanchnic CO2 partial pressure",
          .default_value = 1.0}));
  cardvascrespir0d.specs.emplace_back(parameter<double>("ppO2_arespl_sys_0",
      {.description = "initial systemic arterial extra-splanchnic O2 partial pressure",
          .default_value = 1.0}));
  cardvascrespir0d.specs.emplace_back(parameter<double>("ppCO2_armsc_sys_0",
      {.description = "initial systemic arterial muscular CO2 partial pressure",
          .default_value = 1.0}));
  cardvascrespir0d.specs.emplace_back(parameter<double>(
      "ppO2_armsc_sys_0", {.description = "initial systemic arterial muscular O2 partial pressure",
                              .default_value = 1.0}));
  cardvascrespir0d.specs.emplace_back(parameter<double>("ppCO2_arcer_sys_0",
      {.description = "initial systemic arterial cerebral CO2 partial pressure",
          .default_value = 1.0}));
  cardvascrespir0d.specs.emplace_back(parameter<double>(
      "ppO2_arcer_sys_0", {.description = "initial systemic arterial cerebral O2 partial pressure",
                              .default_value = 1.0}));
  cardvascrespir0d.specs.emplace_back(parameter<double>("ppCO2_arcor_sys_0",
      {.description = "initial systemic arterial coronary CO2 partial pressure",
          .default_value = 1.0}));
  cardvascrespir0d.specs.emplace_back(parameter<double>(
      "ppO2_arcor_sys_0", {.description = "initial systemic arterial coronary O2 partial pressure",
                              .default_value = 1.0}));
  cardvascrespir0d.specs.emplace_back(parameter<double>("ppCO2_venspl_sys_0",
      {.description = "initial systemic venous splanchnic CO2 partial pressure",
          .default_value = 1.0}));
  cardvascrespir0d.specs.emplace_back(parameter<double>(
      "ppO2_venspl_sys_0", {.description = "initial systemic venous splanchnic O2 partial pressure",
                               .default_value = 1.0}));
  cardvascrespir0d.specs.emplace_back(parameter<double>("ppCO2_venespl_sys_0",
      {.description = "initial systemic venous extra-splanchnic CO2 partial pressure",
          .default_value = 1.0}));
  cardvascrespir0d.specs.emplace_back(parameter<double>("ppO2_venespl_sys_0",
      {.description = "initial systemic venous extra-splanchnic O2 partial pressure",
          .default_value = 1.0}));
  cardvascrespir0d.specs.emplace_back(parameter<double>(
      "ppCO2_venmsc_sys_0", {.description = "initial systemic venous muscular CO2 partial pressure",
                                .default_value = 1.0}));
  cardvascrespir0d.specs.emplace_back(parameter<double>(
      "ppO2_venmsc_sys_0", {.description = "initial systemic venous muscular O2 partial pressure",
                               .default_value = 1.0}));
  cardvascrespir0d.specs.emplace_back(parameter<double>(
      "ppCO2_vencer_sys_0", {.description = "initial systemic venous cerebral CO2 partial pressure",
                                .default_value = 1.0}));
  cardvascrespir0d.specs.emplace_back(parameter<double>(
      "ppO2_vencer_sys_0", {.description = "initial systemic venous cerebral O2 partial pressure",
                               .default_value = 1.0}));
  cardvascrespir0d.specs.emplace_back(parameter<double>(
      "ppCO2_vencor_sys_0", {.description = "initial systemic venous coronary CO2 partial pressure",
                                .default_value = 1.0}));
  cardvascrespir0d.specs.emplace_back(parameter<double>(
      "ppO2_vencor_sys_0", {.description = "initial systemic venous coronary O2 partial pressure",
                               .default_value = 1.0}));
  cardvascrespir0d.specs.emplace_back(parameter<double>("ppCO2_ven_sys_0",
      {.description = "initial systemic venous CO2 partial pressure", .default_value = 1.0}));
  cardvascrespir0d.specs.emplace_back(parameter<double>("ppO2_ven_sys_0",
      {.description = "initial systemic venous O2 partial pressure", .default_value = 1.0}));

  cardvascrespir0d.move_into_collection(list);

  Core::Utils::SectionSpecs mor{"MOR"};

  mor.specs.emplace_back(parameter<std::string>("POD_MATRIX",
      {.description = "filename of file containing projection matrix", .default_value = "none"}));

  mor.move_into_collection(list);
}



void Inpar::Cardiovascular0D::set_valid_conditions(
    std::vector<Core::Conditions::ConditionDefinition>& condlist)
{
  using namespace Core::IO::InputSpecBuilders;


  /*--------------------------------------------------------------------*/
  // Monolithic coupling of structure and a four-element windkessel - mhv 11/13

  Core::Conditions::ConditionDefinition cardiovascular0d4elementwindkesselcondition(
      "DESIGN SURF CARDIOVASCULAR 0D 4-ELEMENT WINDKESSEL CONDITIONS",
      "Cardiovascular0D4ElementWindkesselStructureCond", "Surface Cardiovascular0D",
      Core::Conditions::Cardiovascular0D4ElementWindkessel_Structure, true,
      Core::Conditions::geometry_type_surface);

  cardiovascular0d4elementwindkesselcondition.add_component(parameter<int>("id"));
  cardiovascular0d4elementwindkesselcondition.add_component(parameter<double>("C"));
  cardiovascular0d4elementwindkesselcondition.add_component(parameter<double>("R_p"));
  cardiovascular0d4elementwindkesselcondition.add_component(parameter<double>("Z_c"));
  cardiovascular0d4elementwindkesselcondition.add_component(parameter<double>("L"));
  cardiovascular0d4elementwindkesselcondition.add_component(parameter<double>("p_ref"));
  cardiovascular0d4elementwindkesselcondition.add_component(parameter<double>("p_0"));

  condlist.push_back(cardiovascular0d4elementwindkesselcondition);

  /*--------------------------------------------------------------------*/
  // Monolithic coupling of structure and an arterial cardiovascular 0D flow model accounting for
  // proximal and distal arterial pressure formulation proposed by Cristobal Bertoglio - mhv 03/14

  Core::Conditions::ConditionDefinition cardiovascular0darterialproxdistcond(
      "DESIGN SURF CARDIOVASCULAR 0D ARTERIAL PROX DIST CONDITIONS",
      "Cardiovascular0DArterialProxDistStructureCond",
      "Surface 0D cardiovascular arterial proximal and distal",
      Core::Conditions::Cardiovascular0DArterialProxDist_Structure, true,
      Core::Conditions::geometry_type_surface);

  cardiovascular0darterialproxdistcond.add_component(parameter<int>("id"));
  cardiovascular0darterialproxdistcond.add_component(parameter<double>("R_arvalve_max"));
  cardiovascular0darterialproxdistcond.add_component(parameter<double>("R_arvalve_min"));
  cardiovascular0darterialproxdistcond.add_component(parameter<double>("R_atvalve_max"));
  cardiovascular0darterialproxdistcond.add_component(parameter<double>("R_atvalve_min"));
  cardiovascular0darterialproxdistcond.add_component(parameter<double>("k_p"));
  cardiovascular0darterialproxdistcond.add_component(parameter<double>("L_arp"));
  cardiovascular0darterialproxdistcond.add_component(parameter<double>("C_arp"));
  cardiovascular0darterialproxdistcond.add_component(parameter<double>("R_arp"));
  cardiovascular0darterialproxdistcond.add_component(parameter<double>("C_ard"));
  cardiovascular0darterialproxdistcond.add_component(parameter<double>("R_ard"));
  cardiovascular0darterialproxdistcond.add_component(parameter<double>("p_ref"));
  cardiovascular0darterialproxdistcond.add_component(parameter<double>("p_v_0"));
  cardiovascular0darterialproxdistcond.add_component(parameter<double>("p_arp_0"));
  cardiovascular0darterialproxdistcond.add_component(parameter<double>("y_arp_0"));
  cardiovascular0darterialproxdistcond.add_component(parameter<double>("p_ard_0"));
  cardiovascular0darterialproxdistcond.add_component(parameter<double>("p_at_fac"));
  cardiovascular0darterialproxdistcond.add_component(
      parameter<std::optional<int>>("p_at_crv", {.description = "curve"}));
  condlist.push_back(cardiovascular0darterialproxdistcond);

  /*--------------------------------------------------------------------*/
  // Monolithic coupling of structure and a full closed-loop 0D cardiovascular flow model
  // (closed-loop circulatory system model) mhv 02/15

  Core::Conditions::ConditionDefinition cardiovascular0dsyspulcirculationcond(
      "DESIGN SURF CARDIOVASCULAR 0D SYS-PUL CIRCULATION CONDITIONS",
      "Cardiovascular0DSysPulCirculationStructureCond",
      "Surface cardiovascular 0D sys pul circulation condition",
      Core::Conditions::Cardiovascular0DSysPulCirculation_Structure, true,
      Core::Conditions::geometry_type_surface);

  cardiovascular0dsyspulcirculationcond.add_component(parameter<int>("id"));
  cardiovascular0dsyspulcirculationcond.add_component(selection<std::string>("TYPE",
      {"ventricle_left", "ventricle_right", "atrium_left", "atrium_right", "dummy"},
      {.description = ""}));

  condlist.push_back(cardiovascular0dsyspulcirculationcond);

  /*--------------------------------------------------------------------*/
  // Monolithic coupling of structure and a full closed-loop 0D cardiovascular flow model
  // (closed-loop circulatory system model) mhv 02/15

  Core::Conditions::ConditionDefinition cardiovascularrespiratory0dsyspulperiphcirculationcond(
      "DESIGN SURF CARDIOVASCULAR RESPIRATORY 0D SYS-PUL PERIPH CIRCULATION CONDITIONS",
      "CardiovascularRespiratory0DSysPulPeriphCirculationStructureCond",
      "Surface 0D cardiovascular respiratory sys-pul periph circulation condition",
      Core::Conditions::CardiovascularRespiratory0DSysPulPeriphCirculation_Structure, true,
      Core::Conditions::geometry_type_surface);

  cardiovascularrespiratory0dsyspulperiphcirculationcond.add_component(parameter<int>("id"));
  cardiovascularrespiratory0dsyspulperiphcirculationcond.add_component(selection<std::string>(
      "TYPE", {"ventricle_left", "ventricle_right", "atrium_left", "atrium_right", "dummy"},
      {.description = ""}));

  condlist.push_back(cardiovascularrespiratory0dsyspulperiphcirculationcond);


  /*--------------------------------------------------------------------*/
  // Monolithic coupling of structure and 0D cardiovascular flow models: Neumann coupling surface -
  // mhv 11/13

  Core::Conditions::ConditionDefinition cardiovascular0dstructurecouplingcond(
      "DESIGN SURF CARDIOVASCULAR 0D-STRUCTURE COUPLING CONDITIONS",
      "SurfaceNeumannCardiovascular0D", "structure 0d cardiovascular coupling surface condition",
      Core::Conditions::Cardiovascular0DStructureCoupling, true,
      Core::Conditions::geometry_type_surface);

  cardiovascular0dstructurecouplingcond.add_component(parameter<int>("coupling_id"));

  condlist.push_back(cardiovascular0dstructurecouplingcond);
}

FOUR_C_NAMESPACE_CLOSE
