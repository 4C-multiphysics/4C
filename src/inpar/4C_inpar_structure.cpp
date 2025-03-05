// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_inpar_structure.hpp"

#include "4C_constraint_springdashpot.hpp"
#include "4C_fem_condition_definition.hpp"
#include "4C_io_geometry_type.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Inpar
{
  namespace Solid
  {
    void set_valid_parameters(std::map<std::string, Core::IO::InputSpec>& list)
    {
      using Teuchos::tuple;
      using namespace Core::IO::InputSpecBuilders;

      Core::Utils::SectionSpecs sdyn{"STRUCTURAL DYNAMIC"};

      Core::Utils::string_to_integral_parameter<Solid::IntegrationStrategy>("INT_STRATEGY", "Old",
          "global type of the used integration strategy", tuple<std::string>("Old", "Standard"),
          tuple<Solid::IntegrationStrategy>(int_old, int_standard), sdyn);

      sdyn.specs.emplace_back(parameter<bool>("TIME_ADAPTIVITY",
          {.description = "Enable adaptive time integration", .default_value = false}));

      Core::Utils::string_to_integral_parameter<Solid::DynamicType>("DYNAMICTYPE", "GenAlpha",
          "type of the specific dynamic time integration scheme",
          tuple<std::string>("Statics", "GenAlpha", "GenAlphaLieGroup", "OneStepTheta",
              "ExplicitEuler", "CentrDiff", "AdamsBashforth2", "AdamsBashforth4"),
          tuple<Solid::DynamicType>(dyna_statics, dyna_genalpha, dyna_genalpha_liegroup,
              dyna_onesteptheta, dyna_expleuler, dyna_centrdiff, dyna_ab2, dyna_ab4),
          sdyn);

      Core::Utils::string_to_integral_parameter<Inpar::Solid::PreStress>("PRESTRESS", "none",
          "prestressing takes values none mulf material_iterative",
          tuple<std::string>("none", "None", "NONE", "mulf", "Mulf", "MULF", "Material_Iterative",
              "MATERIAL_ITERATIVE", "material_iterative"),
          tuple<Inpar::Solid::PreStress>(Inpar::Solid::PreStress::none,
              Inpar::Solid::PreStress::none, Inpar::Solid::PreStress::none,
              Inpar::Solid::PreStress::mulf, Inpar::Solid::PreStress::mulf,
              Inpar::Solid::PreStress::mulf, Inpar::Solid::PreStress::material_iterative,
              Inpar::Solid::PreStress::material_iterative,
              Inpar::Solid::PreStress::material_iterative),
          sdyn);

      sdyn.specs.emplace_back(parameter<double>("PRESTRESSTIME",
          {.description = "time to switch from pre to post stressing", .default_value = 0.0}));

      sdyn.specs.emplace_back(parameter<double>("PRESTRESSTOLDISP",
          {.description = "tolerance in the displacement norm during prestressing",
              .default_value = 1e-9}));
      Core::Utils::int_parameter(
          "PRESTRESSMINLOADSTEPS", 0, "Minimum number of load steps during prestressing", sdyn);

      // Output type
      Core::Utils::int_parameter("RESULTSEVERY", 1,
          "save displacements and contact forces every RESULTSEVERY steps", sdyn);
      Core::Utils::int_parameter(
          "RESEVERYERGY", 0, "write system energies every requested step", sdyn);
      Core::Utils::int_parameter(
          "RESTARTEVERY", 1, "write restart possibility every RESTARTEVERY steps", sdyn);
      sdyn.specs.emplace_back(parameter<bool>("CALC_ACC_ON_RESTART",
          {.description = "Compute the initial state for a restart dynamics analysis",
              .default_value = false}));
      Core::Utils::int_parameter("OUTPUT_STEP_OFFSET", 0,
          "An offset added to the current step to shift the steps to be written.", sdyn);

      // Time loop control
      sdyn.specs.emplace_back(
          parameter<double>("TIMESTEP", {.description = "time step size", .default_value = 0.05}));
      Core::Utils::int_parameter("NUMSTEP", 200, "maximum number of steps", sdyn);
      sdyn.specs.emplace_back(
          parameter<double>("TIMEINIT", {.description = "initial time", .default_value = 0.0}));
      sdyn.specs.emplace_back(
          parameter<double>("MAXTIME", {.description = "maximum time", .default_value = 5.0}));

      // Damping
      Core::Utils::string_to_integral_parameter<Solid::DampKind>("DAMPING", "None",
          "type of damping: (1) Rayleigh damping matrix and use it from M_DAMP x M + K_DAMP x K, "
          "(2) Material based and calculated in elements",
          tuple<std::string>("None", "Rayleigh", "Material"),
          tuple<Solid::DampKind>(damp_none, damp_rayleigh, damp_material), sdyn);
      sdyn.specs.emplace_back(
          parameter<double>("M_DAMP", {.description = "", .default_value = -1.0}));
      sdyn.specs.emplace_back(
          parameter<double>("K_DAMP", {.description = "", .default_value = -1.0}));

      sdyn.specs.emplace_back(parameter<double>(
          "TOLDISP", {.description = "tolerance in the displacement norm for the newton iteration",
                         .default_value = 1.0E-10}));
      Core::Utils::string_to_integral_parameter<Solid::ConvNorm>("NORM_DISP", "Abs",
          "type of norm for displacement convergence check",
          tuple<std::string>("Abs", "Rel", "Mix"),
          tuple<Solid::ConvNorm>(convnorm_abs, convnorm_rel, convnorm_mix), sdyn);

      sdyn.specs.emplace_back(parameter<double>(
          "TOLRES", {.description = "tolerance in the residual norm for the newton iteration",
                        .default_value = 1.0E-08}));
      Core::Utils::string_to_integral_parameter<Solid::ConvNorm>("NORM_RESF", "Abs",
          "type of norm for residual convergence check", tuple<std::string>("Abs", "Rel", "Mix"),
          tuple<Solid::ConvNorm>(convnorm_abs, convnorm_rel, convnorm_mix), sdyn);

      sdyn.specs.emplace_back(parameter<double>(
          "TOLPRE", {.description = "tolerance in pressure norm for the newton iteration",
                        .default_value = 1.0E-08}));
      Core::Utils::string_to_integral_parameter<Solid::ConvNorm>("NORM_PRES", "Abs",
          "type of norm for pressure convergence check", tuple<std::string>("Abs"),
          tuple<Solid::ConvNorm>(convnorm_abs), sdyn);

      sdyn.specs.emplace_back(parameter<double>("TOLINCO",
          {.description = "tolerance in the incompressible residual norm for the newton iteration",
              .default_value = 1.0E-08}));
      Core::Utils::string_to_integral_parameter<Solid::ConvNorm>("NORM_INCO", "Abs",
          "type of norm for incompressible residual convergence check", tuple<std::string>("Abs"),
          tuple<Solid::ConvNorm>(convnorm_abs), sdyn);

      Core::Utils::string_to_integral_parameter<Solid::BinaryOp>("NORMCOMBI_DISPPRES", "And",
          "binary operator to combine pressure and displacement values",
          tuple<std::string>("And", "Or"), tuple<Solid::BinaryOp>(bop_and, bop_or), sdyn);

      Core::Utils::string_to_integral_parameter<Solid::BinaryOp>("NORMCOMBI_RESFINCO", "And",
          "binary operator to combine force and incompressible residual",
          tuple<std::string>("And", "Or"), tuple<Solid::BinaryOp>(bop_and, bop_or), sdyn);

      Core::Utils::string_to_integral_parameter<Solid::BinaryOp>("NORMCOMBI_RESFDISP", "And",
          "binary operator to combine displacement and residual force values",
          tuple<std::string>("And", "Or"), tuple<Solid::BinaryOp>(bop_and, bop_or), sdyn);

      Core::Utils::string_to_integral_parameter<Solid::StcScale>("STC_SCALING", "Inactive",
          "Scaled director conditioning for thin shell structures",
          tuple<std::string>("Inactive", "Symmetric", "Right"),
          tuple<Solid::StcScale>(stc_inactive, stc_currsym, stc_curr), sdyn);

      Core::Utils::int_parameter("STC_LAYER", 1, "number of STC layers for multilayer case", sdyn);

      sdyn.specs.emplace_back(parameter<double>(
          "PTCDT", {.description = "pseudo time step for pseudo transient continuation (PTC) "
                                   "stabilized Newton procedure",
                       .default_value = 0.1}));

      sdyn.specs.emplace_back(parameter<double>("TOLCONSTR",
          {.description = "tolerance in the constr error norm for the newton iteration",
              .default_value = 1.0E-08}));

      sdyn.specs.emplace_back(parameter<double>("TOLCONSTRINCR",
          {.description = "tolerance in the constr lm incr norm for the newton iteration",
              .default_value = 1.0E-08}));

      Core::Utils::int_parameter("MAXITER", 50,
          "maximum number of iterations allowed for Newton-Raphson iteration before failure", sdyn);
      Core::Utils::int_parameter(
          "MINITER", 0, "minimum number of iterations to be done within Newton-Raphson loop", sdyn);
      Core::Utils::string_to_integral_parameter<Solid::VectorNorm>("ITERNORM", "L2",
          "type of norm to be applied to residuals", tuple<std::string>("L1", "L2", "Rms", "Inf"),
          tuple<Solid::VectorNorm>(norm_l1, norm_l2, norm_rms, norm_inf), sdyn);

      Core::Utils::string_to_integral_parameter<Solid::DivContAct>("DIVERCONT", "stop",
          "What to do with time integration when Newton-Raphson iteration failed",
          tuple<std::string>("stop", "continue", "repeat_step", "halve_step", "adapt_step",
              "rand_adapt_step", "rand_adapt_step_ele_err", "repeat_simulation",
              "adapt_penaltycontact", "adapt_3D0Dptc_ele_err"),
          tuple<Solid::DivContAct>(divcont_stop, divcont_continue, divcont_repeat_step,
              divcont_halve_step, divcont_adapt_step, divcont_rand_adapt_step,
              divcont_rand_adapt_step_ele_err, divcont_repeat_simulation,
              divcont_adapt_penaltycontact, divcont_adapt_3D0Dptc_ele_err),
          sdyn);

      Core::Utils::int_parameter("MAXDIVCONREFINEMENTLEVEL", 10,
          "number of times timestep is halved in case nonlinear solver diverges", sdyn);

      Core::Utils::string_to_integral_parameter<Solid::NonlinSolTech>("NLNSOL", "fullnewton",
          "Nonlinear solution technique",
          tuple<std::string>("vague", "fullnewton", "modnewton", "lsnewton", "ptc",
              "newtonlinuzawa", "augmentedlagrange", "NoxNewtonLineSearch", "noxgeneral", "noxnln",
              "singlestep"),
          tuple<Solid::NonlinSolTech>(soltech_vague, soltech_newtonfull, soltech_newtonmod,
              soltech_newtonls, soltech_ptc, soltech_newtonuzawalin, soltech_newtonuzawanonlin,
              soltech_noxnewtonlinesearch, soltech_noxgeneral, soltech_nox_nln, soltech_singlestep),
          sdyn);

      Core::Utils::int_parameter("LSMAXITER", 30, "maximum number of line search steps", sdyn);
      sdyn.specs.emplace_back(parameter<double>(
          "ALPHA_LS", {.description = "step reduction factor alpha in (Newton) line search scheme",
                          .default_value = 0.5}));
      sdyn.specs.emplace_back(parameter<double>(
          "SIGMA_LS", {.description = "sufficient descent factor in (Newton) line search scheme",
                          .default_value = 1.e-4}));

      std::vector<std::string> material_tangent_valid_input = {"analytical", "finitedifferences"};
      sdyn.specs.emplace_back(
          selection<std::string>("MATERIALTANGENT", material_tangent_valid_input,
              {.description = "way of evaluating the constitutive matrix",
                  .default_value = "analytical"}));


      sdyn.specs.emplace_back(parameter<bool>(
          "LOADLIN", {.description = "Use linearization of external follower load in Newton",
                         .default_value = false}));

      Core::Utils::string_to_integral_parameter<Inpar::Solid::MassLin>("MASSLIN", "none",
          "Application of nonlinear inertia terms", tuple<std::string>("none", "rotations"),
          tuple<Inpar::Solid::MassLin>(
              Inpar::Solid::MassLin::ml_none, Inpar::Solid::MassLin::ml_rotations),
          sdyn);

      sdyn.specs.emplace_back(parameter<bool>(
          "NEGLECTINERTIA", {.description = "Neglect inertia", .default_value = false}));

      // Since predictor "none" would be misleading, the usage of no predictor is called vague.
      Core::Utils::string_to_integral_parameter<Solid::PredEnum>("PREDICT", "ConstDis",
          "Type of predictor",
          tuple<std::string>("Vague", "ConstDis", "ConstVel", "ConstAcc", "ConstDisVelAcc",
              "TangDis", "TangDisConstFext", "ConstDisPres", "ConstDisVelAccPres"),
          tuple<Solid::PredEnum>(pred_vague, pred_constdis, pred_constvel, pred_constacc,
              pred_constdisvelacc, pred_tangdis, pred_tangdis_constfext, pred_constdispres,
              pred_constdisvelaccpres),
          sdyn);

      // Uzawa iteration for constraint systems
      sdyn.specs.emplace_back(parameter<double>("UZAWAPARAM",
          {.description = "Parameter for Uzawa algorithm dealing with lagrange multipliers",
              .default_value = 1.0}));
      sdyn.specs.emplace_back(parameter<double>(
          "UZAWATOL", {.description = "Tolerance for iterative solve with Uzawa algorithm",
                          .default_value = 1.0E-8}));
      Core::Utils::int_parameter("UZAWAMAXITER", 50,
          "maximum number of iterations allowed for uzawa algorithm before failure going to next "
          "newton step",
          sdyn);
      Core::Utils::string_to_integral_parameter<Solid::ConSolveAlgo>("UZAWAALGO", "direct", "",
          tuple<std::string>("uzawa", "simple", "direct"),
          tuple<Solid::ConSolveAlgo>(consolve_uzawa, consolve_simple, consolve_direct), sdyn);

      // convergence criteria adaptivity
      sdyn.specs.emplace_back(parameter<bool>("ADAPTCONV",
          {.description =
                  "Switch on adaptive control of linear solver tolerance for nonlinear solution",
              .default_value = false}));
      sdyn.specs.emplace_back(parameter<double>("ADAPTCONV_BETTER",
          {.description = "The linear solver shall be this much better than the current nonlinear "
                          "residual in the nonlinear convergence limit",
              .default_value = 0.1}));

      sdyn.specs.emplace_back(parameter<bool>(
          "LUMPMASS", {.description = "Lump the mass matrix for explicit time integration",
                          .default_value = false}));

      sdyn.specs.emplace_back(parameter<bool>("MODIFIEDEXPLEULER",
          {.description = "Use the modified explicit Euler time integration scheme",
              .default_value = true}));

      // linear solver id used for structural problems
      Core::Utils::int_parameter(
          "LINEAR_SOLVER", -1, "number of linear solver used for structural problems", sdyn);

      // where the geometry comes from
      Core::Utils::string_to_integral_parameter<Core::IO::GeometryType>("GEOMETRY", "full",
          "How the geometry is specified", tuple<std::string>("full", "box", "file"),
          tuple<Core::IO::GeometryType>(
              Core::IO::geometry_full, Core::IO::geometry_box, Core::IO::geometry_file),
          sdyn);

      Core::Utils::string_to_integral_parameter<Solid::MidAverageEnum>("MIDTIME_ENERGY_TYPE",
          "vague", "Specify the mid-averaging type for the structural energy contributions",
          tuple<std::string>("vague", "imrLike", "trLike"),
          tuple<Solid::MidAverageEnum>(midavg_vague, midavg_imrlike, midavg_trlike), sdyn);

      // Initial displacement
      Core::Utils::string_to_integral_parameter<Solid::InitialDisp>("INITIALDISP",
          "zero_displacement", "Initial displacement for structure problem",
          tuple<std::string>("zero_displacement", "displacement_by_function"),
          tuple<Solid::InitialDisp>(initdisp_zero_disp, initdisp_disp_by_function), sdyn);

      // Function to evaluate initial displacement
      Core::Utils::int_parameter("STARTFUNCNO", -1, "Function for Initial displacement", sdyn);

      sdyn.move_into_collection(list);

      /*--------------------------------------------------------------------*/
      /* parameters for time step size adaptivity in structural dynamics */
      Core::Utils::SectionSpecs tap{sdyn, "TIMEADAPTIVITY"};
      Core::Utils::string_to_integral_parameter<Inpar::Solid::TimAdaKind>("KIND", "None",
          "Method for time step size adaptivity",
          tuple<std::string>("None", "ZienkiewiczXie", "JointExplicit", "AdamsBashforth2",
              "ExplicitEuler", "CentralDifference"),
          tuple<Inpar::Solid::TimAdaKind>(Inpar::Solid::timada_kind_none,
              Inpar::Solid::timada_kind_zienxie, Inpar::Solid::timada_kind_joint_explicit,
              Inpar::Solid::timada_kind_ab2, Inpar::Solid::timada_kind_expleuler,
              Inpar::Solid::timada_kind_centraldiff),
          tap);

      tap.specs.emplace_back(parameter<double>(
          "OUTSYSPERIOD", {.description = "Write system vectors (displacements, velocities, etc) "
                                          "every given period of time",
                              .default_value = 0.0}));
      tap.specs.emplace_back(parameter<double>("OUTSTRPERIOD",
          {.description = "Write stress/strain every given period of time", .default_value = 0.0}));
      tap.specs.emplace_back(parameter<double>("OUTENEPERIOD",
          {.description = "Write energy every given period of time", .default_value = 0.0}));
      tap.specs.emplace_back(parameter<double>("OUTRESTPERIOD",
          {.description = "Write restart data every given period of time", .default_value = 0.0}));
      Core::Utils::int_parameter("OUTSIZEEVERY", 0, "Write step size every given time step", tap);

      tap.specs.emplace_back(parameter<double>("STEPSIZEMAX",
          {.description = "Limit maximally permitted time step size (>0)", .default_value = 0.0}));
      tap.specs.emplace_back(parameter<double>("STEPSIZEMIN",
          {.description = "Limit minimally allowed time step size (>0)", .default_value = 0.0}));
      tap.specs.emplace_back(parameter<double>("SIZERATIOMAX",
          {.description = "Limit maximally permitted change of time step size compared to previous "
                          "size, important for multi-step schemes (>0)",
              .default_value = 0.0}));
      tap.specs.emplace_back(parameter<double>("SIZERATIOMIN",
          {.description = "Limit minimally permitted change of time step size compared to previous "
                          "size, important for multi-step schemes (>0)",
              .default_value = 0.0}));
      tap.specs.emplace_back(parameter<double>("SIZERATIOSCALE",
          {.description = "This is a safety factor to scale theoretical optimal step size, should "
                          "be lower than 1 and must be larger than 0",
              .default_value = 0.9}));

      Core::Utils::string_to_integral_parameter<Inpar::Solid::VectorNorm>("LOCERRNORM", "Vague",
          "Vector norm to treat error vector with",
          tuple<std::string>("Vague", "L1", "L2", "Rms", "Inf"),
          tuple<Inpar::Solid::VectorNorm>(Inpar::Solid::norm_vague, Inpar::Solid::norm_l1,
              Inpar::Solid::norm_l2, Inpar::Solid::norm_rms, Inpar::Solid::norm_inf),
          tap);

      tap.specs.emplace_back(parameter<double>(
          "LOCERRTOL", {.description = "Target local error tolerance (>0)", .default_value = 0.0}));
      Core::Utils::int_parameter(
          "ADAPTSTEPMAX", 0, "Limit maximally allowed step size reduction attempts (>0)", tap);
      tap.move_into_collection(list);

      /// valid parameters for JOINT EXPLICIT

      Core::Utils::SectionSpecs jep{tap, "JOINT EXPLICIT"};

      Core::Utils::int_parameter(
          "LINEAR_SOLVER", -1, "number of linear solver used for auxiliary integrator", jep);

      Core::Utils::string_to_integral_parameter<Inpar::Solid::IntegrationStrategy>("INT_STRATEGY",
          "Standard", "global type of the used integration strategy",
          tuple<std::string>("Standard"), tuple<Inpar::Solid::IntegrationStrategy>(int_standard),
          jep);

      Core::Utils::string_to_integral_parameter<Inpar::Solid::DynamicType>("DYNAMICTYPE",
          "CentrDiff", "type of the specific auxiliary dynamic time integration scheme",
          tuple<std::string>("ExplicitEuler", "CentrDiff", "AdamsBashforth2", "AdamsBashforth4"),
          tuple<Inpar::Solid::DynamicType>(dyna_expleuler, dyna_centrdiff, dyna_ab2, dyna_ab4),
          jep);

      jep.specs.emplace_back(parameter<bool>(
          "LUMPMASS", {.description = "Lump the mass matrix for explicit time integration",
                          .default_value = false}));

      Core::Utils::string_to_integral_parameter<Inpar::Solid::DampKind>("DAMPING", "None",
          "type of damping: (1) Rayleigh damping matrix and use it from M_DAMP x M + K_DAMP x K, "
          "(2) Material based and calculated in elements",
          tuple<std::string>("None", "Rayleigh", "Material"),
          tuple<Inpar::Solid::DampKind>(damp_none, damp_rayleigh, damp_material), jep);

      jep.specs.emplace_back(
          parameter<double>("M_DAMP", {.description = "", .default_value = -1.0}));
      jep.specs.emplace_back(
          parameter<double>("K_DAMP", {.description = "", .default_value = -1.0}));

      jep.move_into_collection(list);

      /*----------------------------------------------------------------------*/
      /* parameters for generalised-alpha structural integrator */
      Core::Utils::SectionSpecs genalpha{sdyn, "GENALPHA"};

      Core::Utils::string_to_integral_parameter<Solid::MidAverageEnum>("GENAVG", "TrLike",
          "mid-average type of internal forces", tuple<std::string>("Vague", "ImrLike", "TrLike"),
          tuple<Solid::MidAverageEnum>(midavg_vague, midavg_imrlike, midavg_trlike), genalpha);
      genalpha.specs.emplace_back(parameter<double>(
          "BETA", {.description = "Generalised-alpha factor in (0,1/2]", .default_value = -1.0}));
      genalpha.specs.emplace_back(parameter<double>(
          "GAMMA", {.description = "Generalised-alpha factor in (0,1]", .default_value = -1.0}));
      genalpha.specs.emplace_back(parameter<double>(
          "ALPHA_M", {.description = "Generalised-alpha factor in [0,1)", .default_value = -1.0}));
      genalpha.specs.emplace_back(parameter<double>(
          "ALPHA_F", {.description = "Generalised-alpha factor in [0,1)", .default_value = -1.0}));
      genalpha.specs.emplace_back(parameter<double>("RHO_INF",
          {.description =
                  "Spectral radius for generalised-alpha time integration, valid range is [0,1]",
              .default_value = 1.0}));

      genalpha.move_into_collection(list);

      /*----------------------------------------------------------------------*/
      /* parameters for one-step-theta structural integrator */
      Core::Utils::SectionSpecs onesteptheta{sdyn, "ONESTEPTHETA"};

      onesteptheta.specs.emplace_back(parameter<double>(
          "THETA", {.description = "One-step-theta factor in (0,1]", .default_value = 0.5}));

      onesteptheta.move_into_collection(list);

      /*----------------------------------------------------------------------*/
      /* parameters for error evaluation */
      Core::Utils::SectionSpecs errorevaluator{sdyn, "ERROR EVALUATION"};
      errorevaluator.specs.emplace_back(parameter<bool>("EVALUATE_ERROR_ANALYTICAL_REFERENCE",
          {.description =
                  "Calculate error with respect to analytical solution defined by a function",
              .default_value = false}));
      Core::Utils::int_parameter("ANALYTICAL_DISPLACEMENT_FUNCTION", -1,
          "function ID of the analytical solution", errorevaluator);

      errorevaluator.move_into_collection(list);
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
        cond.add_component(
            selection<CONSTRAINTS::SpringDashpot::RobinSpringDashpotType>("DIRECTION",
                {{"xyz", CONSTRAINTS::SpringDashpot::RobinSpringDashpotType::xyz},
                    {"refsurfnormal",
                        CONSTRAINTS::SpringDashpot::RobinSpringDashpotType::refsurfnormal},
                    {"cursurfnormal",
                        CONSTRAINTS::SpringDashpot::RobinSpringDashpotType::cursurfnormal}},
                {.description = "Direction of the spring-dashpot boundary conditions"}));
        cond.add_component(parameter<std::optional<int>>("COUPLING", {.description = ""}));
        condlist.emplace_back(cond);
      };

      make_robin_spring_dashpot(robinspringdashpotsurf);
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
