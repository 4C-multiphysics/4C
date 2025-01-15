// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_inpar_structure.hpp"

#include "4C_fem_condition_definition.hpp"
#include "4C_io_geometry_type.hpp"
#include "4C_io_linecomponent.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Inpar
{
  namespace Solid
  {
    /*----------------------------------------------------------------------*/
    /*----------------------------------------------------------------------*/
    void set_valid_time_adaptivity_parameters(Teuchos::ParameterList& list)
    {
      using namespace Input;
      using Teuchos::setStringToIntegralParameter;
      using Teuchos::tuple;

      setStringToIntegralParameter<Inpar::Solid::TimAdaKind>("KIND", "None",
          "Method for time step size adaptivity",
          tuple<std::string>("None", "ZienkiewiczXie", "JointExplicit", "AdamsBashforth2",
              "ExplicitEuler", "CentralDifference"),
          tuple<Inpar::Solid::TimAdaKind>(Inpar::Solid::timada_kind_none,
              Inpar::Solid::timada_kind_zienxie, Inpar::Solid::timada_kind_joint_explicit,
              Inpar::Solid::timada_kind_ab2, Inpar::Solid::timada_kind_expleuler,
              Inpar::Solid::timada_kind_centraldiff),
          &list);

      Core::Utils::double_parameter("OUTSYSPERIOD", 0.0,
          "Write system vectors (displacements, velocities, etc) every given period of time",
          &list);
      Core::Utils::double_parameter(
          "OUTSTRPERIOD", 0.0, "Write stress/strain every given period of time", &list);
      Core::Utils::double_parameter(
          "OUTENEPERIOD", 0.0, "Write energy every given period of time", &list);
      Core::Utils::double_parameter(
          "OUTRESTPERIOD", 0.0, "Write restart data every given period of time", &list);
      Core::Utils::int_parameter("OUTSIZEEVERY", 0, "Write step size every given time step", &list);

      Core::Utils::double_parameter(
          "STEPSIZEMAX", 0.0, "Limit maximally permitted time step size (>0)", &list);
      Core::Utils::double_parameter(
          "STEPSIZEMIN", 0.0, "Limit minimally allowed time step size (>0)", &list);
      Core::Utils::double_parameter("SIZERATIOMAX", 0.0,
          "Limit maximally permitted change of time step size compared to previous size, important "
          "for multi-step schemes (>0)",
          &list);
      Core::Utils::double_parameter("SIZERATIOMIN", 0.0,
          "Limit minimally permitted change of time step size compared to previous size, important "
          "for multi-step schemes (>0)",
          &list);
      Core::Utils::double_parameter("SIZERATIOSCALE", 0.9,
          "This is a safety factor to scale theoretical optimal step size, should be lower than 1 "
          "and must be larger than 0",
          &list);

      setStringToIntegralParameter<Inpar::Solid::VectorNorm>("LOCERRNORM", "Vague",
          "Vector norm to treat error vector with",
          tuple<std::string>("Vague", "L1", "L2", "Rms", "Inf"),
          tuple<Inpar::Solid::VectorNorm>(Inpar::Solid::norm_vague, Inpar::Solid::norm_l1,
              Inpar::Solid::norm_l2, Inpar::Solid::norm_rms, Inpar::Solid::norm_inf),
          &list);

      Core::Utils::double_parameter("LOCERRTOL", 0.0, "Target local error tolerance (>0)", &list);
      Core::Utils::int_parameter(
          "ADAPTSTEPMAX", 0, "Limit maximally allowed step size reduction attempts (>0)", &list);

      /// valid parameters for JOINT EXPLICIT

      Teuchos::ParameterList& jep = list.sublist("JOINT EXPLICIT", false, "");

      Core::Utils::int_parameter(
          "LINEAR_SOLVER", -1, "number of linear solver used for auxiliary integrator", &jep);

      setStringToIntegralParameter<Inpar::Solid::IntegrationStrategy>("INT_STRATEGY", "Standard",
          "global type of the used integration strategy", tuple<std::string>("Standard"),
          tuple<Inpar::Solid::IntegrationStrategy>(int_standard), &jep);

      setStringToIntegralParameter<Inpar::Solid::DynamicType>("DYNAMICTYPE", "CentrDiff",
          "type of the specific auxiliary dynamic time integration scheme",
          tuple<std::string>("ExplicitEuler", "CentrDiff", "AdamsBashforth2", "AdamsBashforth4"),
          tuple<Inpar::Solid::DynamicType>(dyna_expleuler, dyna_centrdiff, dyna_ab2, dyna_ab4),
          &jep);

      Core::Utils::bool_parameter(
          "LUMPMASS", "No", "Lump the mass matrix for explicit time integration", &jep);

      setStringToIntegralParameter<Inpar::Solid::DampKind>("DAMPING", "No",
          "type of damping: (1) Rayleigh damping matrix and use it from M_DAMP x M + K_DAMP x K, "
          "(2) Material based and calculated in elements",
          tuple<std::string>("no", "No", "NO", "yes", "Yes", "YES", "Rayleigh", "Material"),
          tuple<Inpar::Solid::DampKind>(damp_none, damp_none, damp_none, damp_rayleigh,
              damp_rayleigh, damp_rayleigh, damp_rayleigh, damp_material),
          &jep);

      Core::Utils::double_parameter("M_DAMP", -1.0, "", &jep);
      Core::Utils::double_parameter("K_DAMP", -1.0, "", &jep);
    }



    void set_valid_parameters(Teuchos::ParameterList& list)
    {
      using namespace Input;
      using Teuchos::setStringToIntegralParameter;
      using Teuchos::tuple;

      Teuchos::ParameterList& sdyn = list.sublist("STRUCTURAL DYNAMIC", false, "");

      setStringToIntegralParameter<Solid::IntegrationStrategy>("INT_STRATEGY", "Old",
          "global type of the used integration strategy", tuple<std::string>("Old", "Standard"),
          tuple<Solid::IntegrationStrategy>(int_old, int_standard), &sdyn);

      Core::Utils::bool_parameter(
          "TIME_ADAPTIVITY", "No", "Enable adaptive time integration", &sdyn);

      setStringToIntegralParameter<Solid::DynamicType>("DYNAMICTYPE", "GenAlpha",
          "type of the specific dynamic time integration scheme",
          tuple<std::string>("Statics", "GenAlpha", "GenAlphaLieGroup", "OneStepTheta",
              "ExplicitEuler", "CentrDiff", "AdamsBashforth2", "AdamsBashforth4"),
          tuple<Solid::DynamicType>(dyna_statics, dyna_genalpha, dyna_genalpha_liegroup,
              dyna_onesteptheta, dyna_expleuler, dyna_centrdiff, dyna_ab2, dyna_ab4),
          &sdyn);

      setStringToIntegralParameter<Inpar::Solid::PreStress>("PRESTRESS", "none",
          "prestressing takes values none mulf material_iterative",
          tuple<std::string>("none", "None", "NONE", "mulf", "Mulf", "MULF", "Material_Iterative",
              "MATERIAL_ITERATIVE", "material_iterative"),
          tuple<Inpar::Solid::PreStress>(Inpar::Solid::PreStress::none,
              Inpar::Solid::PreStress::none, Inpar::Solid::PreStress::none,
              Inpar::Solid::PreStress::mulf, Inpar::Solid::PreStress::mulf,
              Inpar::Solid::PreStress::mulf, Inpar::Solid::PreStress::material_iterative,
              Inpar::Solid::PreStress::material_iterative,
              Inpar::Solid::PreStress::material_iterative),
          &sdyn);

      Core::Utils::double_parameter(
          "PRESTRESSTIME", 0.0, "time to switch from pre to post stressing", &sdyn);

      Core::Utils::double_parameter("PRESTRESSTOLDISP", 1e-9,
          "tolerance in the displacement norm during prestressing", &sdyn);
      Core::Utils::int_parameter(
          "PRESTRESSMINLOADSTEPS", 0, "Minimum number of load steps during prestressing", &sdyn);

      // Output type
      Core::Utils::int_parameter("RESULTSEVERY", 1,
          "save displacements and contact forces every RESULTSEVERY steps", &sdyn);
      Core::Utils::int_parameter(
          "RESEVERYERGY", 0, "write system energies every requested step", &sdyn);
      Core::Utils::int_parameter(
          "RESTARTEVERY", 1, "write restart possibility every RESTARTEVERY steps", &sdyn);
      Core::Utils::bool_parameter("CALC_ACC_ON_RESTART", "No",
          "Compute the initial state for a restart dynamics analysis", &sdyn);
      Core::Utils::int_parameter("OUTPUT_STEP_OFFSET", 0,
          "An offset added to the current step to shift the steps to be written.", &sdyn);

      // Time loop control
      Core::Utils::double_parameter("TIMESTEP", 0.05, "time step size", &sdyn);
      Core::Utils::int_parameter("NUMSTEP", 200, "maximum number of steps", &sdyn);
      Core::Utils::double_parameter("TIMEINIT", 0.0, "initial time", &sdyn);
      Core::Utils::double_parameter("MAXTIME", 5.0, "maximum time", &sdyn);

      // Damping
      setStringToIntegralParameter<Solid::DampKind>("DAMPING", "No",
          "type of damping: (1) Rayleigh damping matrix and use it from M_DAMP x M + K_DAMP x K, "
          "(2) Material based and calculated in elements",
          tuple<std::string>("no", "No", "NO", "yes", "Yes", "YES", "Rayleigh", "Material"),
          tuple<Solid::DampKind>(damp_none, damp_none, damp_none, damp_rayleigh, damp_rayleigh,
              damp_rayleigh, damp_rayleigh, damp_material),
          &sdyn);
      Core::Utils::double_parameter("M_DAMP", -1.0, "", &sdyn);
      Core::Utils::double_parameter("K_DAMP", -1.0, "", &sdyn);

      Core::Utils::double_parameter(
          "TOLDISP", 1.0E-10, "tolerance in the displacement norm for the newton iteration", &sdyn);
      setStringToIntegralParameter<Solid::ConvNorm>("NORM_DISP", "Abs",
          "type of norm for displacement convergence check",
          tuple<std::string>("Abs", "Rel", "Mix"),
          tuple<Solid::ConvNorm>(convnorm_abs, convnorm_rel, convnorm_mix), &sdyn);

      Core::Utils::double_parameter(
          "TOLRES", 1.0E-08, "tolerance in the residual norm for the newton iteration", &sdyn);
      setStringToIntegralParameter<Solid::ConvNorm>("NORM_RESF", "Abs",
          "type of norm for residual convergence check", tuple<std::string>("Abs", "Rel", "Mix"),
          tuple<Solid::ConvNorm>(convnorm_abs, convnorm_rel, convnorm_mix), &sdyn);

      Core::Utils::double_parameter(
          "TOLPRE", 1.0E-08, "tolerance in pressure norm for the newton iteration", &sdyn);
      setStringToIntegralParameter<Solid::ConvNorm>("NORM_PRES", "Abs",
          "type of norm for pressure convergence check", tuple<std::string>("Abs"),
          tuple<Solid::ConvNorm>(convnorm_abs), &sdyn);

      Core::Utils::double_parameter("TOLINCO", 1.0E-08,
          "tolerance in the incompressible residual norm for the newton iteration", &sdyn);
      setStringToIntegralParameter<Solid::ConvNorm>("NORM_INCO", "Abs",
          "type of norm for incompressible residual convergence check", tuple<std::string>("Abs"),
          tuple<Solid::ConvNorm>(convnorm_abs), &sdyn);

      setStringToIntegralParameter<Solid::BinaryOp>("NORMCOMBI_DISPPRES", "And",
          "binary operator to combine pressure and displacement values",
          tuple<std::string>("And", "Or"), tuple<Solid::BinaryOp>(bop_and, bop_or), &sdyn);

      setStringToIntegralParameter<Solid::BinaryOp>("NORMCOMBI_RESFINCO", "And",
          "binary operator to combine force and incompressible residual",
          tuple<std::string>("And", "Or"), tuple<Solid::BinaryOp>(bop_and, bop_or), &sdyn);

      setStringToIntegralParameter<Solid::BinaryOp>("NORMCOMBI_RESFDISP", "And",
          "binary operator to combine displacement and residual force values",
          tuple<std::string>("And", "Or"), tuple<Solid::BinaryOp>(bop_and, bop_or), &sdyn);

      setStringToIntegralParameter<Solid::StcScale>("STC_SCALING", "no",
          "Scaled director conditioning for thin shell structures",
          tuple<std::string>("no", "No", "NO", "Symmetric", "Right"),
          tuple<Solid::StcScale>(stc_none, stc_none, stc_none, stc_currsym, stc_curr), &sdyn);

      Core::Utils::int_parameter("STC_LAYER", 1, "number of STC layers for multilayer case", &sdyn);

      Core::Utils::double_parameter("PTCDT", 0.1,
          "pseudo time step for pseudo transient continuation (PTC) stabilized Newton procedure",
          &sdyn);

      Core::Utils::double_parameter("TOLCONSTR", 1.0E-08,
          "tolerance in the constr error norm for the newton iteration", &sdyn);

      Core::Utils::double_parameter("TOLCONSTRINCR", 1.0E-08,
          "tolerance in the constr lm incr norm for the newton iteration", &sdyn);

      Core::Utils::int_parameter("MAXITER", 50,
          "maximum number of iterations allowed for Newton-Raphson iteration before failure",
          &sdyn);
      Core::Utils::int_parameter("MINITER", 0,
          "minimum number of iterations to be done within Newton-Raphson loop", &sdyn);
      setStringToIntegralParameter<Solid::VectorNorm>("ITERNORM", "L2",
          "type of norm to be applied to residuals", tuple<std::string>("L1", "L2", "Rms", "Inf"),
          tuple<Solid::VectorNorm>(norm_l1, norm_l2, norm_rms, norm_inf), &sdyn);

      setStringToIntegralParameter<Solid::DivContAct>("DIVERCONT", "stop",
          "What to do with time integration when Newton-Raphson iteration failed",
          tuple<std::string>("stop", "continue", "repeat_step", "halve_step", "adapt_step",
              "rand_adapt_step", "rand_adapt_step_ele_err", "repeat_simulation",
              "adapt_penaltycontact", "adapt_3D0Dptc_ele_err"),
          tuple<Solid::DivContAct>(divcont_stop, divcont_continue, divcont_repeat_step,
              divcont_halve_step, divcont_adapt_step, divcont_rand_adapt_step,
              divcont_rand_adapt_step_ele_err, divcont_repeat_simulation,
              divcont_adapt_penaltycontact, divcont_adapt_3D0Dptc_ele_err),
          &sdyn);

      Core::Utils::int_parameter("MAXDIVCONREFINEMENTLEVEL", 10,
          "number of times timestep is halved in case nonlinear solver diverges", &sdyn);

      setStringToIntegralParameter<Solid::NonlinSolTech>("NLNSOL", "fullnewton",
          "Nonlinear solution technique",
          tuple<std::string>("vague", "fullnewton", "modnewton", "lsnewton", "ptc",
              "newtonlinuzawa", "augmentedlagrange", "NoxNewtonLineSearch", "noxgeneral", "noxnln",
              "singlestep"),
          tuple<Solid::NonlinSolTech>(soltech_vague, soltech_newtonfull, soltech_newtonmod,
              soltech_newtonls, soltech_ptc, soltech_newtonuzawalin, soltech_newtonuzawanonlin,
              soltech_noxnewtonlinesearch, soltech_noxgeneral, soltech_nox_nln, soltech_singlestep),
          &sdyn);

      Core::Utils::int_parameter("LSMAXITER", 30, "maximum number of line search steps", &sdyn);
      Core::Utils::double_parameter(
          "ALPHA_LS", 0.5, "step reduction factor alpha in (Newton) line search scheme", &sdyn);
      Core::Utils::double_parameter(
          "SIGMA_LS", 1.e-4, "sufficient descent factor in (Newton) line search scheme", &sdyn);

      std::vector<std::string> material_tangent_valid_input = {"analytical", "finitedifferences"};
      Core::Utils::string_parameter("MATERIALTANGENT", "analytical",
          "way of evaluating the constitutive matrix", &sdyn, material_tangent_valid_input);


      Core::Utils::bool_parameter(
          "LOADLIN", "No", "Use linearization of external follower load in Newton", &sdyn);

      setStringToIntegralParameter<Solid::MassLin>("MASSLIN", "No",
          "Application of nonlinear inertia terms",
          tuple<std::string>("No", "no", "Standard", "standard", "Rotations", "rotations"),
          tuple<Solid::MassLin>(
              ml_none, ml_none, ml_standard, ml_standard, ml_rotations, ml_rotations),
          &sdyn);

      Core::Utils::bool_parameter("NEGLECTINERTIA", "No", "Neglect inertia", &sdyn);

      // Since predictor "none" would be misleading, the usage of no predictor is called vague.
      setStringToIntegralParameter<Solid::PredEnum>("PREDICT", "ConstDis", "Type of predictor",
          tuple<std::string>("Vague", "ConstDis", "ConstVel", "ConstAcc", "ConstDisVelAcc",
              "TangDis", "TangDisConstFext", "ConstDisPres", "ConstDisVelAccPres"),
          tuple<Solid::PredEnum>(pred_vague, pred_constdis, pred_constvel, pred_constacc,
              pred_constdisvelacc, pred_tangdis, pred_tangdis_constfext, pred_constdispres,
              pred_constdisvelaccpres),
          &sdyn);

      // Uzawa iteration for constraint systems
      Core::Utils::double_parameter("UZAWAPARAM", 1.0,
          "Parameter for Uzawa algorithm dealing with lagrange multipliers", &sdyn);
      Core::Utils::double_parameter(
          "UZAWATOL", 1.0E-8, "Tolerance for iterative solve with Uzawa algorithm", &sdyn);
      Core::Utils::int_parameter("UZAWAMAXITER", 50,
          "maximum number of iterations allowed for uzawa algorithm before failure going to next "
          "newton step",
          &sdyn);
      setStringToIntegralParameter<Solid::ConSolveAlgo>("UZAWAALGO", "direct", "",
          tuple<std::string>("uzawa", "simple", "direct"),
          tuple<Solid::ConSolveAlgo>(consolve_uzawa, consolve_simple, consolve_direct), &sdyn);

      // convergence criteria adaptivity
      Core::Utils::bool_parameter("ADAPTCONV", "No",
          "Switch on adaptive control of linear solver tolerance for nonlinear solution", &sdyn);
      Core::Utils::double_parameter("ADAPTCONV_BETTER", 0.1,
          "The linear solver shall be this much better than the current nonlinear residual in the "
          "nonlinear convergence limit",
          &sdyn);

      Core::Utils::bool_parameter(
          "LUMPMASS", "No", "Lump the mass matrix for explicit time integration", &sdyn);

      Core::Utils::bool_parameter("MODIFIEDEXPLEULER", "Yes",
          "Use the modified explicit Euler time integration scheme", &sdyn);

      // linear solver id used for structural problems
      Core::Utils::int_parameter(
          "LINEAR_SOLVER", -1, "number of linear solver used for structural problems", &sdyn);

      // where the geometry comes from
      setStringToIntegralParameter<Core::IO::GeometryType>("GEOMETRY", "full",
          "How the geometry is specified", tuple<std::string>("full", "box", "file"),
          tuple<Core::IO::GeometryType>(
              Core::IO::geometry_full, Core::IO::geometry_box, Core::IO::geometry_file),
          &sdyn);

      setStringToIntegralParameter<Solid::MidAverageEnum>("MIDTIME_ENERGY_TYPE", "vague",
          "Specify the mid-averaging type for the structural energy contributions",
          tuple<std::string>("vague", "imrLike", "trLike"),
          tuple<Solid::MidAverageEnum>(midavg_vague, midavg_imrlike, midavg_trlike), &sdyn);

      // Initial displacement
      setStringToIntegralParameter<Solid::InitialDisp>("INITIALDISP", "zero_displacement",
          "Initial displacement for structure problem",
          tuple<std::string>("zero_displacement", "displacement_by_function"),
          tuple<Solid::InitialDisp>(initdisp_zero_disp, initdisp_disp_by_function), &sdyn);

      // Function to evaluate initial displacement
      Core::Utils::int_parameter("STARTFUNCNO", -1, "Function for Initial displacement", &sdyn);

      /*--------------------------------------------------------------------*/
      /* parameters for time step size adaptivity in structural dynamics */
      Teuchos::ParameterList& tap = sdyn.sublist("TIMEADAPTIVITY", false, "");
      set_valid_time_adaptivity_parameters(tap);

      /*----------------------------------------------------------------------*/
      /* parameters for generalised-alpha structural integrator */
      Teuchos::ParameterList& genalpha = sdyn.sublist("GENALPHA", false, "");

      setStringToIntegralParameter<Solid::MidAverageEnum>("GENAVG", "TrLike",
          "mid-average type of internal forces", tuple<std::string>("Vague", "ImrLike", "TrLike"),
          tuple<Solid::MidAverageEnum>(midavg_vague, midavg_imrlike, midavg_trlike), &genalpha);
      Core::Utils::double_parameter("BETA", -1.0, "Generalised-alpha factor in (0,1/2]", &genalpha);
      Core::Utils::double_parameter("GAMMA", -1.0, "Generalised-alpha factor in (0,1]", &genalpha);
      Core::Utils::double_parameter(
          "ALPHA_M", -1.0, "Generalised-alpha factor in [0,1)", &genalpha);
      Core::Utils::double_parameter(
          "ALPHA_F", -1.0, "Generalised-alpha factor in [0,1)", &genalpha);
      Core::Utils::double_parameter("RHO_INF", 1.0,
          "Spectral radius for generalised-alpha time integration, valid range is [0,1]",
          &genalpha);

      /*----------------------------------------------------------------------*/
      /* parameters for one-step-theta structural integrator */
      Teuchos::ParameterList& onesteptheta = sdyn.sublist("ONESTEPTHETA", false, "");

      Core::Utils::double_parameter("THETA", 0.5, "One-step-theta factor in (0,1]", &onesteptheta);

      /*----------------------------------------------------------------------*/
      /* parameters for error evaluation */
      Teuchos::ParameterList& errorevaluator = sdyn.sublist("ERROR EVALUATION", false, "");
      Core::Utils::bool_parameter("EVALUATE_ERROR_ANALYTICAL_REFERENCE", "No",
          "Calculate error with respect to analytical solution defined by a function",
          &errorevaluator);
      Core::Utils::int_parameter("ANALYTICAL_DISPLACEMENT_FUNCTION", -1,
          "function ID of the analytical solution", &errorevaluator);
    }



    void set_valid_conditions(
        std::vector<std::shared_ptr<Core::Conditions::ConditionDefinition>>& condlist)
    {
      using namespace Input;

      /*--------------------------------------------------------------------*/

      // structural Robin spring dashpot boundary condition (spring and dashpot in parallel)

      auto robinspringdashpotsurf = std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN SURF ROBIN SPRING DASHPOT CONDITIONS", "RobinSpringDashpot",
          "Robin Spring Dashpot", Core::Conditions::RobinSpringDashpot, true,
          Core::Conditions::geometry_type_surface);

      auto robinspringdashpotpoint = std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN POINT ROBIN SPRING DASHPOT CONDITIONS", "RobinSpringDashpot",
          "Robin Spring Dashpot", Core::Conditions::RobinSpringDashpot, true,
          Core::Conditions::geometry_type_point);

      for (const auto& cond : {robinspringdashpotpoint, robinspringdashpotsurf})
      {
        add_named_int(cond, "NUMDOF");
        add_named_int_vector(cond, "ONOFF", "", 3);
        add_named_real_vector(cond, "STIFF", "", 3);
        add_named_int_vector(cond, "TIMEFUNCTSTIFF", "", 3);
        add_named_real_vector(cond, "VISCO", "", 3);
        add_named_int_vector(cond, "TIMEFUNCTVISCO", "", 3);
        add_named_real_vector(cond, "DISPLOFFSET", "", 3);
        add_named_int_vector(cond, "TIMEFUNCTDISPLOFFSET", "", 3);
        add_named_int_vector(cond, "FUNCTNONLINSTIFF", "", 3);
        add_named_selection_component(cond, "DIRECTION", "", "xyz",
            Teuchos::tuple<std::string>("xyz", "refsurfnormal", "cursurfnormal"),
            Teuchos::tuple<std::string>("xyz", "refsurfnormal", "cursurfnormal"), false);
        add_named_int(cond, "COUPLING", "", 0, false);
        condlist.emplace_back(cond);
      }

      /*--------------------------------------------------------------------*/
      // surface coupling for spring dashpot DIRECTION cursurfnormal
      // pfaller Apr15

      std::shared_ptr<Core::Conditions::ConditionDefinition> springdashpotcoupcond =
          std::make_shared<Core::Conditions::ConditionDefinition>(
              "DESIGN SURF ROBIN SPRING DASHPOT COUPLING CONDITIONS", "RobinSpringDashpotCoupling",
              "RobinSpring Dashpot Coupling", Core::Conditions::RobinSpringDashpotCoupling, true,
              Core::Conditions::geometry_type_surface);

      add_named_int(springdashpotcoupcond, "COUPLING");

      condlist.push_back(springdashpotcoupcond);
    }
  }  // end of namespace Solid
}  // end of namespace Inpar

FOUR_C_NAMESPACE_CLOSE
