// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_red_airways_input.hpp"

#include "4C_fem_condition_definition.hpp"
#include "4C_global_data.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_mat_maxwell_0d_acinus.hpp"
#include "4C_red_airways_elementbase.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
| Read in the RED_AIRWAY elements                                       |
*-----------------------------------------------------------------------*/
bool Discret::Elements::RedAirway::read_element(const std::string& eletype,
    const std::string& distype, const Core::IO::InputParameterContainer& container,
    const Core::IO::MeshInput::ElementDataFromCellData& element_data)
{
  const int ndim = Global::Problem::instance()->n_dim();
  if (ndim != 3)
    FOUR_C_THROW("Problem defined as {}d, but found Reduced dimensional AIRWAY element.", ndim);

  // Read number of material model
  int material_id = container.get<int>("MAT");
  set_material(0, Mat::factory(material_id));

  // Read the element type, the element specific variables and store them to airwayParams_
  elem_type_ = container.get<std::string>("TYPE");
  resistance_ = container.get<std::string>("Resistance");
  elemsolving_type_ = container.get<std::string>("ElemSolvingType");

  double velPow = container.get<double>("PowerOfVelocityProfile");
  double Ew = container.get<double>("WallElasticity");
  double nu = container.get<double>("PoissonsRatio");
  double Ts = container.get<double>("ViscousTs");
  double Phis = container.get<double>("ViscousPhaseShift");
  double tw = container.get<double>("WallThickness");
  double A = container.get<double>("Area");
  int generation = container.get<int>("Generation");

  if (container.get<std::optional<double>>("AirwayColl").has_value())
  {
    airway_params_.airway_coll = *container.get<std::optional<double>>("AirwayColl");
    airway_params_.s_close = *container.get<std::optional<double>>("S_Close");
    airway_params_.s_open = *container.get<std::optional<double>>("S_Open");
    airway_params_.p_crit_open = *container.get<std::optional<double>>("Pcrit_Open");
    airway_params_.p_crit_close = *container.get<std::optional<double>>("Pcrit_Close");
    airway_params_.open_init = *container.get<std::optional<double>>("Open_Init");
  }

  // Correct the velocity profile power
  // this is because the 2.0 is the minimum energy consumtive laminar profile
  if (velPow < 2.0) velPow = 2.0;
  airway_params_.power_velocity_profile = velPow;
  airway_params_.wall_elasticity = Ew;
  airway_params_.poisson_ratio = nu;
  airway_params_.wall_thickness = tw;
  airway_params_.area = A;
  airway_params_.viscous_Ts = Ts;
  airway_params_.viscous_phase_shift = Phis;
  airway_params_.generation = generation;
  airway_params_.branch_length = container.get<std::optional<double>>("BranchLength").value_or(-1);

  return true;
}


/*----------------------------------------------------------------------*
| Read in the RED_ACINUS elements                                       |
*-----------------------------------------------------------------------*/
bool Discret::Elements::RedAcinus::read_element(const std::string& eletype,
    const std::string& distype, const Core::IO::InputParameterContainer& container,
    const Core::IO::MeshInput::ElementDataFromCellData& element_data)
{
  const int ndim = Global::Problem::instance()->n_dim();
  if (ndim != 3)
    FOUR_C_THROW("Problem defined as {}d, but found Reduced dimensional ACINUS element.", ndim);

  // Read number of material model
  int material_id = container.get<int>("MAT");
  set_material(0, Mat::factory(material_id));

  // Read the element type, the element specific variables and store them to acinusParams_
  elem_type_ = container.get<std::string>("TYPE");

  acinus_params_.volume_relaxed = container.get<double>("AcinusVolume");
  acinus_params_.alveolar_duct_volume = container.get<double>("AlveolarDuctVolume");
  acinus_params_.volume_init = acinus_params_.volume_relaxed;
  acinus_params_.generation = -1;

  // Setup material, calls overloaded function setup(linedef) for each Maxwell_0d_acinus material
  std::shared_ptr<Core::Mat::Material> mat = material();
  std::shared_ptr<Mat::Maxwell0dAcinus> acinus_mat =
      std::dynamic_pointer_cast<Mat::Maxwell0dAcinus>(material());
  acinus_mat->setup(container);

  return true;
}


/*----------------------------------------------------------------------*
| Read in the RED_ACINAR_INTER_DEP elements                             |
*-----------------------------------------------------------------------*/
bool Discret::Elements::RedInterAcinarDep::read_element(const std::string& eletype,
    const std::string& distype, const Core::IO::InputParameterContainer& container,
    const Core::IO::MeshInput::ElementDataFromCellData& element_data)
{
  const int ndim = Global::Problem::instance()->n_dim();
  if (ndim != 3)
    FOUR_C_THROW(
        "Problem defined as {}d, but found Reduced dimensional INTER ACINAR DEPENDENCE element.",
        ndim);

  // set generation
  const int generation = -2;
  generation_ = generation;

  // Read number of material model
  int material_id = container.get<int>("MAT");
  set_material(0, Mat::factory(material_id));


  return true;
}

Core::IO::InputSpec Airway::valid_parameters()
{
  using namespace Core::IO::InputSpecBuilders;
  Core::IO::InputSpec spec = group("REDUCED DIMENSIONAL AIRWAYS DYNAMIC",
      {

          parameter<Airway::RedAirwaysDyntype>(
              "DYNAMICTYPE", {.description = "Dynamic Type",
                                 .default_value = Airway::RedAirwaysDyntype::OneStepTheta}),

          parameter<Airway::RedAirwaysSolvertype>(
              "SOLVERTYPE", {.description = "Solver Type",
                                .default_value = Airway::RedAirwaysSolvertype::Linear}),

          parameter<double>(
              "TIMESTEP", {.description = "Time increment dt", .default_value = 0.01}),

          parameter<int>("NUMSTEP", {.description = "Number of Time Steps", .default_value = 0}),
          parameter<int>(
              "RESTARTEVERY", {.description = "Increment for writing restart", .default_value = 1}),
          parameter<int>("RESULTSEVERY",
              {.description = "Increment for writing solution", .default_value = 1}),
          parameter<double>("THETA",
              {.description = "One-step-theta time integration factor", .default_value = 1.0}),

          parameter<int>(
              "MAXITERATIONS", {.description = "maximum iteration steps", .default_value = 1}),

          parameter<double>("TOLERANCE", {.description = "tolerance", .default_value = 1.0E-6}),

          // number of linear solver used for reduced dimensional airways dynamic
          parameter<int>("LINEAR_SOLVER",
              {.description = "number of linear solver used for reduced dim arterial dynamics",
                  .default_value = -1}),

          parameter<bool>("SOLVESCATRA",
              {.description = "Flag to (de)activate solving scalar transport in blood",
                  .default_value = false}),

          parameter<bool>("COMPAWACINTER",
              {.description = "Flag to (de)activate computation of airway-acinus interdependency",
                  .default_value = false}),

          parameter<bool>(
              "CALCV0PRESTRESS", {.description = "Flag to (de)activate initial acini volume "
                                                 "adjustment with pre-stress condition",
                                     .default_value = false}),

          parameter<double>("TRANSPULMPRESS",
              {.description = "Transpulmonary pressure needed for recalculation of acini volumes",
                  .default_value = 800.0})},
      {.required = false});
  return spec;
}



void Airway::set_valid_conditions(std::vector<Core::Conditions::ConditionDefinition>& condlist)
{
  using namespace Core::IO::InputSpecBuilders;

  /*--------------------------------------------------------------------*/
  // 3-D/reduced-D coupling boundary condition
  Core::Conditions::ConditionDefinition art_red_to_3d_bc(
      "DESIGN NODE REDUCED D To 3D FLOW COUPLING CONDITIONS", "Art_redD_3D_CouplingCond",
      "Artery reduced D 3D coupling condition", Core::Conditions::ArtRedTo3DCouplingCond, true,
      Core::Conditions::geometry_type_point);

  art_red_to_3d_bc.add_component(parameter<int>("ConditionID"));
  art_red_to_3d_bc.add_component(deprecated_selection<std::string>("CouplingType",
      {"forced", "absorbing"}, {.description = "coupling type", .default_value = "forced"}));
  art_red_to_3d_bc.add_component(deprecated_selection<std::string>("ReturnedVariable",
      {"pressure", "flow"}, {.description = "returned variable", .default_value = "pressure"}));
  art_red_to_3d_bc.add_component(parameter<double>("Tolerance"));
  art_red_to_3d_bc.add_component(parameter<int>("MaximumIterations"));

  condlist.push_back(art_red_to_3d_bc);

  /*--------------------------------------------------------------------*/
  // 3-D/reduced-D coupling boundary condition
  Core::Conditions::ConditionDefinition art_3d_to_red_bc(
      "DESIGN SURF 3D To REDUCED D FLOW COUPLING CONDITIONS", "Art_3D_redD_CouplingCond",
      "Artery 3D reduced D coupling condition", Core::Conditions::Art3DToRedCouplingCond, true,
      Core::Conditions::geometry_type_surface);

  art_3d_to_red_bc.add_component(parameter<int>("ConditionID"));
  art_3d_to_red_bc.add_component(deprecated_selection<std::string>("ReturnedVariable",
      {"pressure", "flow"}, {.description = "returned variable", .default_value = "flow"}));
  art_3d_to_red_bc.add_component(parameter<double>("Tolerance"));
  art_3d_to_red_bc.add_component(parameter<int>("MaximumIterations"));

  condlist.push_back(art_3d_to_red_bc);


  /*--------------------------------------------------------------------*/
  // Prescribed BC for reduced dimensional airways

  Core::Conditions::ConditionDefinition raw_in_bc(
      "DESIGN NODE Reduced D AIRWAYS PRESCRIBED CONDITIONS", "RedAirwayPrescribedCond",
      "Reduced d airway prescribed boundary condition", Core::Conditions::RedAirwayPrescribedCond,
      true, Core::Conditions::geometry_type_point);

  raw_in_bc.add_component(deprecated_selection<std::string>("boundarycond",
      {"flow", "pressure", "switchFlowPressure", "VolumeDependentPleuralPressure"},
      {.description = "boundary condition type", .default_value = "flow"}));

  // reduced airway inlet components
  raw_in_bc.add_component(
      parameter<std::vector<double>>("VAL", {.description = "value", .size = 1}));
  raw_in_bc.add_component(
      parameter<std::vector<std::optional<int>>>("curve", {.description = "curve", .size = 2}));
  raw_in_bc.add_component(parameter<std::vector<std::optional<int>>>(
      "funct", {.description = "function id",
                   .default_value = std::vector{std::optional<int>{}},
                   .size = 1}));

  condlist.push_back(raw_in_bc);

  /*--------------------------------------------------------------------*/
  // Prescribed BC for reduced dimensional airways switching between different types of boundary
  // conditions

  Core::Conditions::ConditionDefinition raw_in_switch_bc(
      "DESIGN NODE Reduced D AIRWAYS SWITCH FLOW PRESSURE CONDITIONS",
      "RedAirwaySwitchFlowPressureCond", "Reduced d airway switch flow pressure boundary condition",
      Core::Conditions::RedAirwayPrescribedSwitchCond, true, Core::Conditions::geometry_type_point);

  raw_in_switch_bc.add_component(parameter<int>("FUNCT_ID_FLOW"));
  raw_in_switch_bc.add_component(parameter<int>("FUNCT_ID_PRESSURE"));
  raw_in_switch_bc.add_component(parameter<int>("FUNCT_ID_PRESSURE_ACTIVE"));

  condlist.push_back(raw_in_switch_bc);

  /*--------------------------------------------------------------------*/
  // Prescribed volume dependent pleural pressure for reduced dimensional airways

  Core::Conditions::ConditionDefinition raw_volPpl_bc(
      "DESIGN LINE REDUCED D AIRWAYS VOL DEPENDENT PLEURAL PRESSURE CONDITIONS",
      "RedAirwayVolDependentPleuralPressureCond",
      "Reduced D airways volume-dependent peural pressure condition",
      Core::Conditions::RedAirwayVolDependentPleuralPressureCond, true,
      Core::Conditions::geometry_type_line);

  raw_volPpl_bc.add_component(deprecated_selection<std::string>("TYPE",
      {"Linear_Polynomial", "Linear_Exponential", "Linear_Ogden", "Nonlinear_Polynomial",
          "Nonlinear_Exponential", "Nonlinear_Ogden"},
      {.description = "type", .default_value = "Linear_Exponential"}));

  raw_volPpl_bc.add_component(parameter<double>("TLC"));
  raw_volPpl_bc.add_component(parameter<double>("RV"));

  raw_volPpl_bc.add_component(parameter<double>("P_PLEURAL_0"));
  raw_volPpl_bc.add_component(parameter<double>("P_PLEURAL_LIN"));
  raw_volPpl_bc.add_component(parameter<double>("P_PLEURAL_NONLIN"));
  raw_volPpl_bc.add_component(parameter<double>("TAU"));

  // raw_volPpl_bc_components
  raw_volPpl_bc.add_component(
      parameter<std::vector<double>>("VAL", {.description = "value", .size = 1}));
  raw_volPpl_bc.add_component(
      parameter<std::vector<std::optional<int>>>("curve", {.description = "curve", .size = 1}));

  condlist.push_back(raw_volPpl_bc);

  /*--------------------------------------------------------------------*/
  // Evaluate lung volume condition for reduced dimensional airways

  Core::Conditions::ConditionDefinition raw_eval_lungV_bc(
      "DESIGN LINE REDUCED D AIRWAYS EVALUATE LUNG VOLUME CONDITIONS", "RedAirwayEvalLungVolCond",
      "Reduced D airways evaluate lung volume condition",
      Core::Conditions::RedAirwayEvalLungVolCond, true, Core::Conditions::geometry_type_line);


  condlist.push_back(raw_eval_lungV_bc);


  /*--------------------------------------------------------------------*/
  // Impedance condition

  Core::Conditions::ConditionDefinition impedancebc("DESIGN SURF IMPEDANCE CONDITIONS",
      "ImpedanceCond", "Impedance boundary condition", Core::Conditions::ImpedanceCond, true,
      Core::Conditions::geometry_type_surface);

  impedancebc.add_component(parameter<int>("ConditionID"));
  impedancebc.add_component(
      deprecated_selection<std::string>("TYPE", {"windkessel", "resistive", "pressure_by_funct"},
          {.description = "type", .default_value = "windkessel"}));
  impedancebc.add_component(parameter<double>("R1"));
  impedancebc.add_component(parameter<double>("R2"));
  impedancebc.add_component(parameter<double>("C"));
  impedancebc.add_component(parameter<double>("TIMEPERIOD"));
  impedancebc.add_component(parameter<int>("FUNCT"));

  condlist.push_back(impedancebc);
}


FOUR_C_NAMESPACE_CLOSE
