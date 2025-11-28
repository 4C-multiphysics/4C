// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_inpar_structure.hpp"

#include "4C_constraint_springdashpot.hpp"
#include "4C_fem_condition_definition.hpp"
#include "4C_io_input_spec.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_utils_enum.hpp"
FOUR_C_NAMESPACE_OPEN

namespace Inpar
{
  namespace Solid
  {
    std::string pred_enum_string(const PredEnum name)
    {
      switch (name)
      {
        case pred_vague:
          return "Vague";
        case pred_constdis:
          return "ConstDis";
        case pred_constvel:
          return "ConstVel";
        case pred_constacc:
          return "ConstAcc";
        case pred_constdisvelacc:
          return "ConstDisVelAcc";
        case pred_tangdis:
          return "TangDis";
        case pred_tangdis_constfext:
          return "TangDisConstFext";
        case pred_constdispres:
          return "ConstDisPres";
        case pred_constdisvelaccpres:
          return "ConstDisVelAccPres";
        default:
          FOUR_C_THROW("Cannot make std::string for predictor {}", name);
      }
    }

    std::string kinem_type_string(const KinemType kinem_type)
    {
      switch (kinem_type)
      {
        case Inpar::Solid::KinemType::vague:
          return "vague";
        case Inpar::Solid::KinemType::linear:
          return "linear";
        case Inpar::Solid::KinemType::nonlinearTotLag:
          return "nonlinear";
      }

      FOUR_C_THROW("Unknown kinematic type {}", kinem_type);
    }

    std::vector<Core::IO::InputSpec> valid_parameters()
    {
      using namespace Core::IO::InputSpecBuilders;

      std::vector<Core::IO::InputSpec> specs;
      specs.push_back(group("STRUCTURAL DYNAMIC",
          {

              deprecated_selection<Solid::IntegrationStrategy>("INT_STRATEGY",
                  {
                      {"Old", int_old},
                      {"Standard", int_standard},
                  },
                  {.description = "global type of the used integration strategy",
                      .default_value = int_old}),

              parameter<bool>("TIME_ADAPTIVITY",
                  {.description = "Enable adaptive time integration", .default_value = false}),

              deprecated_selection<Solid::DynamicType>("DYNAMICTYPE",
                  {
                      {"Statics", DynamicType::Statics},
                      {"GenAlpha", DynamicType::GenAlpha},
                      {"GenAlphaLieGroup", DynamicType::GenAlphaLieGroup},
                      {"OneStepTheta", DynamicType::OneStepTheta},
                      {"ExplicitEuler", DynamicType::ExplEuler},
                      {"CentrDiff", DynamicType::CentrDiff},
                      {"AdamsBashforth2", DynamicType::AdamsBashforth2},
                      {"AdamsBashforth4", DynamicType::AdamsBashforth4},
                  },
                  {.description = "type of the specific dynamic time integration scheme",
                      .default_value = DynamicType::GenAlpha}),

              deprecated_selection<Inpar::Solid::PreStress>("PRESTRESS",
                  {
                      {"none", Inpar::Solid::PreStress::none},
                      {"None", Inpar::Solid::PreStress::none},
                      {"NONE", Inpar::Solid::PreStress::none},
                      {"mulf", Inpar::Solid::PreStress::mulf},
                      {"Mulf", Inpar::Solid::PreStress::mulf},
                      {"MULF", Inpar::Solid::PreStress::mulf},
                      {"Material_Iterative", Inpar::Solid::PreStress::material_iterative},
                      {"MATERIAL_ITERATIVE", Inpar::Solid::PreStress::material_iterative},
                      {"material_iterative", Inpar::Solid::PreStress::material_iterative},
                  },
                  {.description = "prestressing takes values none mulf material_iterative",
                      .default_value = Inpar::Solid::PreStress::none}),

              parameter<double>(
                  "PRESTRESSTIME", {.description = "time to switch from pre to post stressing",
                                       .default_value = 0.0}),

              parameter<double>("PRESTRESSTOLDISP",
                  {.description = "tolerance in the displacement norm during prestressing",
                      .default_value = 1e-9}),
              parameter<int>("PRESTRESSMINLOADSTEPS",
                  {.description = "Minimum number of load steps during prestressing",
                      .default_value = 0}),

              // Output type
              // TODO: Modify docu once Contact is migrated to new vtk-based output
              parameter<int>("RESULTSEVERY",
                  {.description = "Write old HDF5-based output every RESULTSEVERY steps. Used for "
                                  "old structure time integration and Contact quantities.",
                      .default_value = 0}),
              parameter<int>(
                  "RESEVERYERGY", {.description = "write system energies every requested step",
                                      .default_value = 0}),
              parameter<int>("RESTARTEVERY",
                  {.description = "write restart possibility every RESTARTEVERY steps",
                      .default_value = 0}),
              parameter<bool>("CALC_ACC_ON_RESTART",
                  {.description = "Compute the initial state for a restart dynamics analysis",
                      .default_value = false}),
              parameter<int>("OUTPUT_STEP_OFFSET",
                  {.description =
                          "An offset added to the current step to shift the steps to be written.",
                      .default_value = 0}),

              // Time loop control
              parameter<double>(
                  "TIMESTEP", {.description = "time step size", .default_value = 0.05}),
              parameter<int>(
                  "NUMSTEP", {.description = "maximum number of steps", .default_value = 200}),

              parameter<double>("TIMEINIT", {.description = "initial time", .default_value = 0.0}),

              parameter<double>("MAXTIME", {.description = "maximum time", .default_value = 5.0}),

              // Damping
              deprecated_selection<Solid::DampKind>("DAMPING",
                  {
                      {"None", damp_none},
                      {"Rayleigh", damp_rayleigh},
                      {"Material", damp_material},
                  },
                  {.description =
                          "type of damping: (1) Rayleigh damping matrix and use it from M_DAMP x M "
                          "+ K_DAMP x K, (2) Material based and calculated in elements",
                      .default_value = damp_none}),

              parameter<double>("M_DAMP", {.description = "", .default_value = -1.0}),

              parameter<double>("K_DAMP", {.description = "", .default_value = -1.0}),

              parameter<double>("TOLDISP",
                  {.description = "tolerance in the displacement norm for the newton iteration",
                      .default_value = 1.0E-10}),
              deprecated_selection<Solid::ConvNorm>("NORM_DISP",
                  {
                      {"Abs", convnorm_abs},
                      {"Rel", convnorm_rel},
                      {"Mix", convnorm_mix},
                  },
                  {.description = "type of norm for displacement convergence check",
                      .default_value = convnorm_abs}),

              parameter<double>("TOLRES",
                  {.description = "tolerance in the residual norm for the newton iteration",
                      .default_value = 1.0E-08}),
              deprecated_selection<Solid::ConvNorm>("NORM_RESF",
                  {
                      {"Abs", convnorm_abs},
                      {"Rel", convnorm_rel},
                      {"Mix", convnorm_mix},
                  },
                  {.description = "type of norm for residual convergence check",
                      .default_value = convnorm_abs}),

              parameter<double>(
                  "TOLPRE", {.description = "tolerance in pressure norm for the newton iteration",
                                .default_value = 1.0E-08}),
              deprecated_selection<Solid::ConvNorm>("NORM_PRES",
                  {
                      {"Abs", convnorm_abs},
                  },
                  {.description = "type of norm for pressure convergence check",
                      .default_value = convnorm_abs}),

              parameter<double>("TOLINCO",
                  {.description =
                          "tolerance in the incompressible residual norm for the newton iteration",
                      .default_value = 1.0E-08}),
              deprecated_selection<Solid::ConvNorm>("NORM_INCO",
                  {
                      {"Abs", convnorm_abs},
                  },
                  {.description = "type of norm for incompressible residual convergence check",
                      .default_value = convnorm_abs}),

              deprecated_selection<Solid::BinaryOp>("NORMCOMBI_DISPPRES",
                  {
                      {"And", bop_and},
                      {"Or", bop_or},
                  },
                  {.description = "binary operator to combine pressure and displacement values",
                      .default_value = bop_and}),

              deprecated_selection<Solid::BinaryOp>("NORMCOMBI_RESFINCO",
                  {
                      {"And", bop_and},
                      {"Or", bop_or},
                  },
                  {.description = "binary operator to combine force and incompressible residual",
                      .default_value = bop_and}),

              deprecated_selection<Solid::BinaryOp>("NORMCOMBI_RESFDISP",
                  {
                      {"And", bop_and},
                      {"Or", bop_or},
                  },
                  {.description =
                          "binary operator to combine displacement and residual force values",
                      .default_value = bop_and}),

              deprecated_selection<Solid::StcScale>("STC_SCALING",
                  {
                      {"Inactive", stc_inactive},
                      {"Symmetric", stc_currsym},
                      {"Right", stc_curr},
                  },
                  {.description = "Scaled director conditioning for thin shell structures",
                      .default_value = stc_inactive}),

              parameter<int>("STC_LAYER",
                  {.description = "number of STC layers for multilayer case", .default_value = 1}),

              parameter<double>("PTCDT",
                  {.description = "pseudo time step for pseudo transient continuation (PTC) "
                                  "stabilized Newton procedure",
                      .default_value = 0.1}),

              parameter<double>("TOLCONSTR",
                  {.description = "tolerance in the constr error norm for the newton iteration",
                      .default_value = 1.0E-08}),

              parameter<double>("TOLCONSTRINCR",
                  {.description = "tolerance in the constr lm incr norm for the newton iteration",
                      .default_value = 1.0E-08}),

              parameter<int>("MAXITER",
                  {.description = "maximum number of iterations allowed for Newton-Raphson "
                                  "iteration before failure",
                      .default_value = 50}),
              parameter<int>("MINITER",
                  {.description =
                          "minimum number of iterations to be done within Newton-Raphson loop",
                      .default_value = 0}),
              deprecated_selection<Solid::VectorNorm>("ITERNORM",
                  {
                      {"L1", norm_l1},
                      {"L2", norm_l2},
                      {"Rms", norm_rms},
                      {"Inf", norm_inf},
                  },
                  {.description = "type of norm to be applied to residuals",
                      .default_value = norm_l2}),

              deprecated_selection<Solid::DivContAct>("DIVERCONT",
                  {
                      {"stop", divcont_stop},
                      {"continue", divcont_continue},
                      {"repeat_step", divcont_repeat_step},
                      {"halve_step", divcont_halve_step},
                      {"adapt_step", divcont_adapt_step},
                      {"rand_adapt_step", divcont_rand_adapt_step},
                      {"rand_adapt_step_ele_err", divcont_rand_adapt_step_ele_err},
                      {"repeat_simulation", divcont_repeat_simulation},
                      {"adapt_penaltycontact", divcont_adapt_penaltycontact},
                      {"adapt_3D0Dptc_ele_err", divcont_adapt_3D0Dptc_ele_err},
                  },
                  {.description =
                          "What to do with time integration when Newton-Raphson iteration failed",
                      .default_value = divcont_stop}),

              parameter<int>("MAXDIVCONREFINEMENTLEVEL",
                  {.description =
                          "number of times timestep is halved in case nonlinear solver diverges",
                      .default_value = 10}),

              deprecated_selection<Solid::NonlinSolTech>("NLNSOL",
                  {
                      {"vague", soltech_vague},
                      {"fullnewton", soltech_newtonfull},
                      {"modnewton", soltech_newtonmod},
                      {"lsnewton", soltech_newtonls},
                      {"ptc", soltech_ptc},
                      {"newtonlinuzawa", soltech_newtonuzawalin},
                      {"augmentedlagrange", soltech_newtonuzawanonlin},
                      {"noxnln", soltech_nox_nln},
                      {"singlestep", soltech_singlestep},
                  },
                  {.description = "Nonlinear solution technique",
                      .default_value = soltech_newtonfull}),

              parameter<int>("LSMAXITER",
                  {.description = "maximum number of line search steps", .default_value = 30}),
              parameter<double>("ALPHA_LS",
                  {.description = "step reduction factor alpha in (Newton) line search scheme",
                      .default_value = 0.5}),
              parameter<double>("SIGMA_LS",
                  {.description = "sufficient descent factor in (Newton) line search scheme",
                      .default_value = 1.e-4}),

              deprecated_selection<std::string>("MATERIALTANGENT",
                  {"analytical", "finitedifferences"},
                  {.description = "way of evaluating the constitutive matrix",
                      .default_value = "analytical"}),


              parameter<bool>("LOADLIN",
                  {.description = "Use linearization of external follower load in Newton",
                      .default_value = false}),

              deprecated_selection<Inpar::Solid::MassLin>("MASSLIN",
                  {
                      {"none", Inpar::Solid::MassLin::ml_none},
                      {"rotations", Inpar::Solid::MassLin::ml_rotations},
                  },
                  {.description = "Application of nonlinear inertia terms",
                      .default_value = Inpar::Solid::MassLin::ml_none}),

              parameter<bool>(
                  "NEGLECTINERTIA", {.description = "Neglect inertia", .default_value = false}),

              // Since predictor "none" would be misleading, the usage of no predictor is called
              // vague.
              deprecated_selection<Solid::PredEnum>("PREDICT",
                  {
                      {"Vague", pred_vague},
                      {"ConstDis", pred_constdis},
                      {"ConstVel", pred_constvel},
                      {"ConstAcc", pred_constacc},
                      {"ConstDisVelAcc", pred_constdisvelacc},
                      {"TangDis", pred_tangdis},
                      {"TangDisConstFext", pred_tangdis_constfext},
                      {"ConstDisPres", pred_constdispres},
                      {"ConstDisVelAccPres", pred_constdisvelaccpres},
                  },
                  {.description = "Type of predictor", .default_value = pred_constdis}),

              // Uzawa iteration for constraint systems
              parameter<double>("UZAWAPARAM",
                  {.description = "Parameter for Uzawa algorithm dealing with lagrange multipliers",
                      .default_value = 1.0}),
              parameter<double>(
                  "UZAWATOL", {.description = "Tolerance for iterative solve with Uzawa algorithm",
                                  .default_value = 1.0E-8}),
              parameter<int>("UZAWAMAXITER",
                  {.description = "maximum number of iterations allowed for uzawa "
                                  "algorithm before failure going to next newton step",
                      .default_value = 50}),
              deprecated_selection<Solid::ConSolveAlgo>("UZAWAALGO",
                  {
                      {"uzawa", consolve_uzawa},
                      {"simple", consolve_simple},
                      {"direct", consolve_direct},
                  },
                  {.description = "", .default_value = consolve_direct}),

              // convergence criteria adaptivity
              parameter<bool>(
                  "ADAPTCONV", {.description = "Switch on adaptive control of linear "
                                               "solver tolerance for nonlinear solution",
                                   .default_value = false}),
              parameter<double>("ADAPTCONV_BETTER",
                  {.description =
                          "The linear solver shall be this much better than the current nonlinear "
                          "residual in the nonlinear convergence limit",
                      .default_value = 0.1}),

              parameter<bool>(
                  "LUMPMASS", {.description = "Lump the mass matrix for explicit time integration",
                                  .default_value = false}),

              parameter<bool>("MODIFIEDEXPLEULER",
                  {.description = "Use the modified explicit Euler time integration scheme",
                      .default_value = true}),

              // linear solver id used for structural problems
              parameter<int>("LINEAR_SOLVER",
                  {.description = "number of linear solver used for structural problems",
                      .default_value = -1}),

              deprecated_selection<Solid::MidAverageEnum>("MIDTIME_ENERGY_TYPE",
                  {
                      {"vague", midavg_vague},
                      {"imrLike", midavg_imrlike},
                      {"trLike", midavg_trlike},
                  },
                  {.description =
                          "Specify the mid-averaging type for the structural energy contributions",
                      .default_value = midavg_vague}),

              // Initial displacement
              deprecated_selection<Solid::InitialDisp>("INITIALDISP",
                  {
                      {"zero_displacement", initdisp_zero_disp},
                      {"displacement_by_function", initdisp_disp_by_function},
                  },
                  {.description = "Initial displacement for structure problem",
                      .default_value = initdisp_zero_disp}),

              // Function to evaluate initial displacement
              parameter<int>("STARTFUNCNO",
                  {.description = "Function for Initial displacement", .default_value = -1})},
          {.required =
                  false})); /*--------------------------------------------------------------------*/
      /* parameters for time step size adaptivity in structural dynamics */
      specs.push_back(group("STRUCTURAL DYNAMIC/TIMEADAPTIVITY",
          {

              deprecated_selection<Inpar::Solid::TimAdaKind>("KIND",
                  {
                      {"None", Inpar::Solid::timada_kind_none},
                      {"ZienkiewiczXie", Inpar::Solid::timada_kind_zienxie},
                      {"JointExplicit", Inpar::Solid::timada_kind_joint_explicit},
                      {"AdamsBashforth2", Inpar::Solid::timada_kind_ab2},
                      {"ExplicitEuler", Inpar::Solid::timada_kind_expleuler},
                      {"CentralDifference", Inpar::Solid::timada_kind_centraldiff},
                  },
                  {.description = "Method for time step size adaptivity",
                      .default_value = Inpar::Solid::timada_kind_none}),

              parameter<double>("OUTSYSPERIOD",
                  {.description = "Write system vectors (displacements, velocities, etc) "
                                  "every given period of time",
                      .default_value = 0.0}),
              parameter<double>(
                  "OUTSTRPERIOD", {.description = "Write stress/strain every given period of time",
                                      .default_value = 0.0}),
              parameter<double>("OUTENEPERIOD",
                  {.description = "Write energy every given period of time", .default_value = 0.0}),
              parameter<double>(
                  "OUTRESTPERIOD", {.description = "Write restart data every given period of time",
                                       .default_value = 0.0}),
              parameter<int>("OUTSIZEEVERY",
                  {.description = "Write step size every given time step", .default_value = 0}),

              parameter<double>(
                  "STEPSIZEMAX", {.description = "Limit maximally permitted time step size (>0)",
                                     .default_value = 0.0}),
              parameter<double>(
                  "STEPSIZEMIN", {.description = "Limit minimally allowed time step size (>0)",
                                     .default_value = 0.0}),
              parameter<double>("SIZERATIOMAX",
                  {.description =
                          "Limit maximally permitted change of time step size compared to previous "
                          "size, important for multi-step schemes (>0)",
                      .default_value = 0.0}),
              parameter<double>("SIZERATIOMIN",
                  {.description =
                          "Limit minimally permitted change of time step size compared to previous "
                          "size, important for multi-step schemes (>0)",
                      .default_value = 0.0}),
              parameter<double>("SIZERATIOSCALE",
                  {.description =
                          "This is a safety factor to scale theoretical optimal step size, should "
                          "be lower than 1 and must be larger than 0",
                      .default_value = 0.9}),

              deprecated_selection<Inpar::Solid::VectorNorm>("LOCERRNORM",
                  {
                      {"Vague", Inpar::Solid::norm_vague},
                      {"L1", Inpar::Solid::norm_l1},
                      {"L2", Inpar::Solid::norm_l2},
                      {"Rms", Inpar::Solid::norm_rms},
                      {"Inf", Inpar::Solid::norm_inf},
                  },
                  {.description = "Vector norm to treat error vector with",
                      .default_value = Inpar::Solid::norm_vague}),

              parameter<double>("LOCERRTOL",
                  {.description = "Target local error tolerance (>0)", .default_value = 0.0}),
              parameter<int>("ADAPTSTEPMAX",
                  {.description = "Limit maximally allowed step size reduction attempts (>0)",
                      .default_value = 0})},
          {.required = false}));

      /// valid parameters for JOINT EXPLICIT

      specs.push_back(group("STRUCTURAL DYNAMIC/TIMEADAPTIVITY/JOINT EXPLICIT",
          {

              parameter<int>("LINEAR_SOLVER",
                  {.description = "number of linear solver used for auxiliary integrator",
                      .default_value = -1}),

              deprecated_selection<Inpar::Solid::IntegrationStrategy>("INT_STRATEGY",
                  {
                      {"Standard", int_standard},
                  },
                  {.description = "global type of the used integration strategy",
                      .default_value = int_standard}),

              deprecated_selection<Inpar::Solid::DynamicType>("DYNAMICTYPE",
                  {
                      {"ExplicitEuler", DynamicType::ExplEuler},
                      {"CentrDiff", DynamicType::CentrDiff},
                      {"AdamsBashforth2", DynamicType::AdamsBashforth2},
                      {"AdamsBashforth4", DynamicType::AdamsBashforth4},
                  },
                  {.description = "type of the specific auxiliary dynamic time integration scheme",
                      .default_value = DynamicType::CentrDiff}),

              parameter<bool>(
                  "LUMPMASS", {.description = "Lump the mass matrix for explicit time integration",
                                  .default_value = false}),

              deprecated_selection<Inpar::Solid::DampKind>("DAMPING",
                  {
                      {"None", damp_none},
                      {"Rayleigh", damp_rayleigh},
                      {"Material", damp_material},
                  },
                  {.description =
                          "type of damping: (1) Rayleigh damping matrix and use it from M_DAMP x M "
                          "+ K_DAMP x K, (2) Material based and calculated in elements",
                      .default_value = damp_none}),


              parameter<double>("M_DAMP", {.description = "", .default_value = -1.0}),

              parameter<double>("K_DAMP", {.description = "", .default_value = -1.0})},
          {.required = false}));

      /*----------------------------------------------------------------------*/
      /* parameters for generalised-alpha structural integrator */
      specs.push_back(group("STRUCTURAL DYNAMIC/GENALPHA",
          {

              deprecated_selection<Solid::MidAverageEnum>("GENAVG",
                  {
                      {"Vague", midavg_vague},
                      {"ImrLike", midavg_imrlike},
                      {"TrLike", midavg_trlike},
                  },
                  {.description = "mid-average type of internal forces",
                      .default_value = midavg_trlike}),
              parameter<double>("BETA",
                  {.description = "Generalised-alpha factor in (0,1/2]", .default_value = -1.0}),
              parameter<double>("GAMMA",
                  {.description = "Generalised-alpha factor in (0,1]", .default_value = -1.0}),
              parameter<double>("ALPHA_M",
                  {.description = "Generalised-alpha factor in [0,1)", .default_value = -1.0}),
              parameter<double>("ALPHA_F",
                  {.description = "Generalised-alpha factor in [0,1)", .default_value = -1.0}),
              parameter<double>(
                  "RHO_INF", {.description = "Spectral radius for generalised-alpha time "
                                             "integration, valid range is [0,1]",
                                 .default_value = 1.0})},
          {.required = false}));

      /*----------------------------------------------------------------------*/
      /* parameters for one-step-theta structural integrator */
      specs.push_back(group("STRUCTURAL DYNAMIC/ONESTEPTHETA",
          {

              parameter<double>("THETA",
                  {.description = "One-step-theta factor in (0,1]", .default_value = 0.5})},
          {.required = false}));

      /*----------------------------------------------------------------------*/
      /* parameters for error evaluation */
      specs.push_back(group("STRUCTURAL DYNAMIC/ERROR EVALUATION",
          {

              parameter<bool>("EVALUATE_ERROR_ANALYTICAL_REFERENCE",
                  {.description = "Calculate error with respect to analytical solution defined by "
                                  "a function",
                      .default_value = false}),
              parameter<int>("ANALYTICAL_DISPLACEMENT_FUNCTION",
                  {.description = "function ID of the analytical solution", .default_value = -1})},
          {.required = false}));
      return specs;
    }



    void set_valid_conditions(std::vector<Core::Conditions::ConditionDefinition>& condlist)
    {
      using namespace Core::IO::InputSpecBuilders;

      /*--------------------------------------------------------------------*/

      // structural Robin spring dashpot boundary condition (spring and dashpot in parallel)

      Core::Conditions::ConditionDefinition robinspringdashpotsurf(
          "DESIGN SURF ROBIN SPRING DASHPOT CONDITIONS", "RobinSpringDashpot",
          "Robin Spring Dashpot", Core::Conditions::RobinSpringDashpot, true,
          Core::Conditions::geometry_type_surface);


      Core::Conditions::ConditionDefinition robinspringdashpotline(
          "DESIGN LINE ROBIN SPRING DASHPOT CONDITIONS", "RobinSpringDashpot",
          "Robin Spring Dashpot", Core::Conditions::RobinSpringDashpot, true,
          Core::Conditions::geometry_type_line);

      Core::Conditions::ConditionDefinition robinspringdashpotpoint(
          "DESIGN POINT ROBIN SPRING DASHPOT CONDITIONS", "RobinSpringDashpot",
          "Robin Spring Dashpot", Core::Conditions::RobinSpringDashpot, true,
          Core::Conditions::geometry_type_point);

      const auto make_robin_spring_dashpot = [&condlist](
                                                 Core::Conditions::ConditionDefinition& cond)
      {
        cond.add_component(parameter<int>("NUMDOF"));
        cond.add_component(parameter<std::vector<int>>(
            "ONOFF", {.description = "", .size = from_parameter<int>("NUMDOF")}));
        cond.add_component(parameter<std::vector<double>>(
            "STIFF", {.description = "", .size = from_parameter<int>("NUMDOF")}));
        cond.add_component(parameter<std::vector<int>>(
            "TIMEFUNCTSTIFF", {.description = "", .size = from_parameter<int>("NUMDOF")}));
        cond.add_component(parameter<std::vector<double>>(
            "VISCO", {.description = "", .size = from_parameter<int>("NUMDOF")}));
        cond.add_component(parameter<std::vector<int>>(
            "TIMEFUNCTVISCO", {.description = "", .size = from_parameter<int>("NUMDOF")}));
        cond.add_component(parameter<std::vector<double>>(
            "DISPLOFFSET", {.description = "", .size = from_parameter<int>("NUMDOF")}));
        cond.add_component(parameter<std::vector<int>>(
            "TIMEFUNCTDISPLOFFSET", {.description = "", .size = from_parameter<int>("NUMDOF")}));
        cond.add_component(parameter<std::vector<int>>(
            "FUNCTNONLINSTIFF", {.description = "", .size = from_parameter<int>("NUMDOF")}));
        cond.add_component(parameter<Constraints::SpringDashpot::RobinSpringDashpotType>(
            "DIRECTION", {.description = "Direction of the spring-dashpot boundary conditions"}));
        cond.add_component(parameter<std::optional<int>>("COUPLING", {.description = ""}));
        condlist.emplace_back(cond);
      };

      make_robin_spring_dashpot(robinspringdashpotsurf);
      make_robin_spring_dashpot(robinspringdashpotline);
      make_robin_spring_dashpot(robinspringdashpotpoint);

      /*--------------------------------------------------------------------*/
      // surface coupling for spring dashpot DIRECTION cursurfnormal
      // pfaller Apr15

      Core::Conditions::ConditionDefinition springdashpotcoupcond(
          "DESIGN SURF ROBIN SPRING DASHPOT COUPLING CONDITIONS", "RobinSpringDashpotCoupling",
          "RobinSpring Dashpot Coupling", Core::Conditions::RobinSpringDashpotCoupling, true,
          Core::Conditions::geometry_type_surface);

      springdashpotcoupcond.add_component(parameter<int>("COUPLING"));

      condlist.push_back(springdashpotcoupcond);
    }
  }  // end of namespace Solid
}  // end of namespace Inpar

FOUR_C_NAMESPACE_CLOSE