// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_inpar_solver_nonlin.hpp"

#include "4C_solver_nonlin_nox_enum_lists.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Inpar::NlnSol::set_valid_parameters(std::map<std::string, Core::IO::InputSpec>& list)
{
  using Teuchos::tuple;
  using namespace Core::IO::InputSpecBuilders;

  /*----------------------------------------------------------------------*
   * parameters for NOX - non-linear solution
   *----------------------------------------------------------------------*/
  Core::Utils::SectionSpecs snox{"STRUCT NOX"};

  {
    std::vector<std::string> nonlinear_solver_valid_input = {"Line Search Based",
        "Pseudo Transient", "Trust Region Based", "Inexact Trust Region Based", "Tensor Based",
        "Single Step"};

    snox.specs.emplace_back(
        deprecated_selection<std::string>("Nonlinear Solver", nonlinear_solver_valid_input,
            {.description = "Choose a nonlinear solver method.",
                .default_value = "Line Search Based"}));
  }
  snox.move_into_collection(list);

  // sub-list direction
  Core::Utils::SectionSpecs direction{snox, "Direction"};

  {
    std::vector<std::string> newton_method_valid_input = {
        "Newton", "Steepest Descent", "NonlinearCG", "Broyden", "User Defined"};
    direction.specs.emplace_back(
        deprecated_selection<std::string>("Method", newton_method_valid_input,
            {.description = "Choose a direction method for the nonlinear solver.",
                .default_value = "Newton"}));

    std::vector<std::string> user_defined_method_valid_input = {"Newton", "Modified Newton"};
    direction.specs.emplace_back(
        deprecated_selection<std::string>("User Defined Method", user_defined_method_valid_input,
            {.description = "Choose a user-defined direction method.",
                .default_value = "Modified Newton"}));
  }
  direction.move_into_collection(list);

  // sub-sub-list "Newton"
  Core::Utils::SectionSpecs newton{direction, "Newton"};

  {
    std::vector<std::string> forcing_term_valid_input = {"Constant", "Type 1", "Type 2"};
    newton.specs.emplace_back(deprecated_selection<std::string>("Forcing Term Method",
        forcing_term_valid_input, {.description = "", .default_value = "Constant"}));

    newton.specs.emplace_back(parameter<double>("Forcing Term Initial Tolerance",
        {.description = "initial linear solver tolerance", .default_value = 0.1}));
    newton.specs.emplace_back(parameter<double>(
        "Forcing Term Minimum Tolerance", {.description = "", .default_value = 1.0e-6}));
    newton.specs.emplace_back(parameter<double>(
        "Forcing Term Maximum Tolerance", {.description = "", .default_value = 0.01}));
    Core::Utils::double_parameter("Forcing Term Alpha", 1.5, "used only by \"Type 2\"", newton);
    Core::Utils::double_parameter("Forcing Term Gamma", 0.9, "used only by \"Type 2\"", newton);
    newton.specs.emplace_back(parameter<bool>("Rescue Bad Newton Solve",
        {.description = "If set to true, we will use the computed direction even if the linear "
                        "solve does not achieve the tolerance specified by the forcing term",
            .default_value = true}));
  }
  newton.move_into_collection(list);

  // sub-sub-list "Steepest Descent"
  Core::Utils::SectionSpecs steepestdescent{direction, "Steepest Descent"};

  {
    std::vector<std::string> scaling_type_valid_input = {
        "2-Norm", "Quadratic Model Min", "F 2-Norm", "None"};
    steepestdescent.specs.emplace_back(deprecated_selection<std::string>(
        "Scaling Type", scaling_type_valid_input, {.description = "", .default_value = "None"}));
  }
  steepestdescent.move_into_collection(list);

  // sub-list "Pseudo Transient"
  Core::Utils::SectionSpecs ptc{snox, "Pseudo Transient"};

  {
    ptc.specs.emplace_back(parameter<double>(
        "deltaInit", {.description = "Initial time step size. If its negative, the initial time "
                                     "step is calculated automatically.",
                         .default_value = -1.0}));
    Core::Utils::double_parameter("deltaMax", std::numeric_limits<double>::max(),
        "Maximum time step size. "
        "If the new step size is greater than this value, the transient terms will be eliminated "
        "from the Newton iteration resulting in a full Newton solve.",
        ptc);
    ptc.specs.emplace_back(parameter<double>(
        "deltaMin", {.description = "Minimum step size.", .default_value = 1.0e-5}));
    Core::Utils::int_parameter(
        "Max Number of PTC Iterations", std::numeric_limits<int>::max(), "", ptc);
    ptc.specs.emplace_back(
        parameter<double>("SER_alpha", {.description = "Exponent of SET.", .default_value = 1.0}));
    ptc.specs.emplace_back(parameter<double>(
        "ScalingFactor", {.description = "Scaling Factor for ptc matrix.", .default_value = 1.0}));

    std::vector<std::string> time_step_control_valid_input = {"SER",
        "Switched Evolution Relaxation", "TTE", "Temporal Truncation Error", "MRR",
        "Model Reduction Ratio"};
    ptc.specs.emplace_back(deprecated_selection<std::string>("Time Step Control",
        time_step_control_valid_input, {.description = "", .default_value = "SER"}));

    std::vector<std::string> tsc_norm_type_valid_input = {"Two Norm", "One Norm", "Max Norm"};
    ptc.specs.emplace_back(
        deprecated_selection<std::string>("Norm Type for TSC", tsc_norm_type_valid_input,
            {.description = "Norm Type for the time step control", .default_value = "Max Norm"}));

    std::vector<std::string> scaling_op_valid_input = {
        "Identity", "CFL Diagonal", "Lumped Mass", "Element based"};
    ptc.specs.emplace_back(deprecated_selection<std::string>("Scaling Type", scaling_op_valid_input,
        {.description = "Type of the scaling matrix for the PTC method.",
            .default_value = "Identity"}));

    std::vector<std::string> build_scale_op_valid_input = {"every iter", "every timestep"};
    ptc.specs.emplace_back(
        deprecated_selection<std::string>("Build scaling operator", build_scale_op_valid_input,
            {.description = "Build scaling operator in every iteration or timestep",
                .default_value = "every timestep"}));
  }
  ptc.move_into_collection(list);

  // sub-list "Line Search"
  Core::Utils::SectionSpecs linesearch{snox, "Line Search"};

  {
    std::vector<std::string> method_valid_input = {
        "Full Step", "Backtrack", "Polynomial", "More'-Thuente", "User Defined"};
    linesearch.specs.emplace_back(deprecated_selection<std::string>(
        "Method", method_valid_input, {.description = "", .default_value = "Full Step"}));


    Teuchos::Array<std::string> checktypes =
        Teuchos::tuple<std::string>("Complete", "Minimal", "None");
    Core::Utils::string_to_integral_parameter<::NOX::StatusTest::CheckType>(
        "Inner Status Test Check Type", "Minimal",
        "Specify the check type for the inner status tests.", checktypes,
        Teuchos::tuple<::NOX::StatusTest::CheckType>(
            ::NOX::StatusTest::Complete, ::NOX::StatusTest::Minimal, ::NOX::StatusTest::None),
        linesearch);
  }
  linesearch.move_into_collection(list);

  // sub-sub-list "Full Step"
  Core::Utils::SectionSpecs fullstep{linesearch, "Full Step"};

  {
    fullstep.specs.emplace_back(parameter<double>(
        "Full Step", {.description = "length of a full step", .default_value = 1.0}));
  }
  fullstep.move_into_collection(list);

  // sub-sub-list "Backtrack"
  Core::Utils::SectionSpecs backtrack{linesearch, "Backtrack"};

  {
    backtrack.specs.emplace_back(parameter<double>(
        "Default Step", {.description = "starting step length", .default_value = 1.0}));
    backtrack.specs.emplace_back(parameter<double>("Minimum Step",
        {.description = "minimum acceptable step length", .default_value = 1.0e-12}));
    Core::Utils::double_parameter("Recovery Step", 1.0,
        "step to take when the line search fails (defaults to value for \"Default Step\")",
        backtrack);
    Core::Utils::int_parameter(
        "Max Iters", 50, "maximum number of iterations (i.e., RHS computations)", backtrack);
    backtrack.specs.emplace_back(parameter<double>(
        "Reduction Factor", {.description = "A multiplier between zero and one that reduces the "
                                            "step size between line search iterations",
                                .default_value = 0.5}));
    backtrack.specs.emplace_back(parameter<bool>("Allow Exceptions",
        {.description = "Set to true, if exceptions during the force evaluation and backtracking "
                        "routine should be allowed.",
            .default_value = false}));
  }
  backtrack.move_into_collection(list);

  // sub-sub-list "Polynomial"
  Core::Utils::SectionSpecs polynomial{linesearch, "Polynomial"};

  {
    polynomial.specs.emplace_back(parameter<double>(
        "Default Step", {.description = "Starting step length", .default_value = 1.0}));
    Core::Utils::int_parameter("Max Iters", 100,
        "Maximum number of line search iterations. "
        "The search fails if the number of iterations exceeds this value",
        polynomial);
    polynomial.specs.emplace_back(parameter<double>(
        "Minimum Step", {.description = "Minimum acceptable step length. The search fails if the "
                                        "computed $\\lambda_k$ is less than this value",
                            .default_value = 1.0e-12}));

    std::vector<std::string> recovery_step_type_valid_input = {"Constant", "Last Computed Step"};
    polynomial.specs.emplace_back(
        deprecated_selection<std::string>("Recovery Step Type", recovery_step_type_valid_input,
            {.description = "Determines the step size to take when the line search fails",
                .default_value = "Constant"}));

    Core::Utils::double_parameter("Recovery Step", 1.0,
        "The value of the step to take when the line search fails. Only used if the \"Recovery "
        "Step Type\" is set to \"Constant\"",
        polynomial);

    std::vector<std::string> interpolation_type_valid_input = {"Quadratic", "Quadratic3", "Cubic"};
    polynomial.specs.emplace_back(deprecated_selection<std::string>("Interpolation Type",
        interpolation_type_valid_input,
        {.description = "Type of interpolation that should be used", .default_value = "Cubic"}));

    polynomial.specs.emplace_back(parameter<double>("Min Bounds Factor",
        {.description = "Choice for $\\gamma_{\\min}$, i.e., the factor that limits the minimum "
                        "size of the new step based on the previous step",
            .default_value = 0.1}));
    polynomial.specs.emplace_back(parameter<double>("Max Bounds Factor",
        {.description = "Choice for $\\gamma_{\\max}$, i.e., the factor that limits the maximum "
                        "size of the new step based on the previous step",
            .default_value = 0.5}));

    std::vector<std::string> sufficient_decrease_condition_valid_input = {
        "Armijo-Goldstein", "Ared/Pred", "None"};
    polynomial.specs.emplace_back(deprecated_selection<std::string>("Sufficient Decrease Condition",
        sufficient_decrease_condition_valid_input,
        {.description = "Choice to use for the sufficient decrease condition",
            .default_value = "Armijo-Goldstein"}));

    polynomial.specs.emplace_back(parameter<double>(
        "Alpha Factor", {.description = "Parameter choice for sufficient decrease condition",
                            .default_value = 1.0e-4}));
    polynomial.specs.emplace_back(parameter<bool>("Force Interpolation",
        {.description = "Set to true if at least one interpolation step should be used. The "
                        "default is false which means that the line search will stop if the "
                        "default step length satisfies the convergence criteria",
            .default_value = false}));
    polynomial.specs.emplace_back(parameter<bool>("Use Counters",
        {.description = "Set to true if we should use counters and then output the result to the "
                        "parameter list as described in Output Parameters",
            .default_value = true}));
    Core::Utils::int_parameter("Maximum Iteration for Increase", 0,
        "Maximum index of the nonlinear iteration for which we allow a relative increase",
        polynomial);
    polynomial.specs.emplace_back(parameter<double>(
        "Allowed Relative Increase", {.description = "", .default_value = 100.0}));
  }
  polynomial.move_into_collection(list);

  // sub-sub-list "More'-Thuente"
  Core::Utils::SectionSpecs morethuente{linesearch, "More'-Thuente"};

  {
    morethuente.specs.emplace_back(parameter<double>("Sufficient Decrease",
        {.description = "The ftol in the sufficient decrease condition", .default_value = 1.0e-4}));
    morethuente.specs.emplace_back(parameter<double>("Curvature Condition",
        {.description = "The gtol in the curvature condition", .default_value = 0.9999}));
    morethuente.specs.emplace_back(parameter<double>("Interval Width",
        {.description =
                "The maximum width of the interval containing the minimum of the modified function",
            .default_value = 1.0e-15}));
    morethuente.specs.emplace_back(parameter<double>(
        "Maximum Step", {.description = "maximum allowable step length", .default_value = 1.0e6}));
    morethuente.specs.emplace_back(parameter<double>("Minimum Step",
        {.description = "minimum allowable step length", .default_value = 1.0e-12}));
    Core::Utils::int_parameter("Max Iters", 20,
        "maximum number of right-hand-side and corresponding Jacobian evaluations", morethuente);
    morethuente.specs.emplace_back(parameter<double>(
        "Default Step", {.description = "starting step length", .default_value = 1.0}));

    std::vector<std::string> recovery_step_type_valid_input = {"Constant", "Last Computed Step"};
    morethuente.specs.emplace_back(
        deprecated_selection<std::string>("Recovery Step Type", recovery_step_type_valid_input,
            {.description = "Determines the step size to take when the line search fails",
                .default_value = "Constant"}));

    Core::Utils::double_parameter("Recovery Step", 1.0,
        "The value of the step to take when the line search fails. Only used if the \"Recovery "
        "Step Type\" is set to \"Constant\"",
        morethuente);

    std::vector<std::string> sufficient_decrease_condition_valid_input = {
        "Armijo-Goldstein", "Ared/Pred", "None"};
    morethuente.specs.emplace_back(deprecated_selection<std::string>(
        "Sufficient Decrease Condition", sufficient_decrease_condition_valid_input,
        {.description = "Choice to use for the sufficient decrease condition",
            .default_value = "Armijo-Goldstein"}));

    morethuente.specs.emplace_back(parameter<bool>("Optimize Slope Calculation",
        {.description = "Boolean value. If set to true the value of $s^T J^T F$ is estimated using "
                        "a directional derivative in a call to "
                        "::NOX::LineSearch::Common::computeSlopeWithOutJac. If false the slope "
                        "computation is computed with the ::NOX::LineSearch::Common::computeSlope "
                        "method. Setting this to true eliminates having to compute the Jacobian at "
                        "each inner iteration of the More'-Thuente line search",
            .default_value = false}));
  }
  morethuente.move_into_collection(list);

  // sub-list "Trust Region"
  Core::Utils::SectionSpecs trustregion{snox, "Trust Region"};

  {
    trustregion.specs.emplace_back(parameter<double>("Minimum Trust Region Radius",
        {.description = "Minimum allowable trust region radius", .default_value = 1.0e-6}));
    trustregion.specs.emplace_back(parameter<double>("Maximum Trust Region Radius",
        {.description = "Maximum allowable trust region radius", .default_value = 1.0e+9}));
    trustregion.specs.emplace_back(parameter<double>("Minimum Improvement Ratio",
        {.description = "Minimum improvement ratio to accept the step", .default_value = 1.0e-4}));
    Core::Utils::double_parameter("Contraction Trigger Ratio", 0.1,
        "If the improvement ratio is less than this value, then the trust region is contracted by "
        "the amount specified by the \"Contraction Factor\". Must be larger than \"Minimum "
        "Improvement Ratio\"",
        trustregion);
    trustregion.specs.emplace_back(
        parameter<double>("Contraction Factor", {.description = "", .default_value = 0.25}));
    Core::Utils::double_parameter("Expansion Trigger Ratio", 0.75,
        "If the improvement ratio is greater than this value, then the trust region is contracted "
        "by the amount specified by the \"Expansion Factor\"",
        trustregion);
    trustregion.specs.emplace_back(
        parameter<double>("Expansion Factor", {.description = "", .default_value = 4.0}));
    trustregion.specs.emplace_back(
        parameter<double>("Recovery Step", {.description = "", .default_value = 1.0}));
  }
  trustregion.move_into_collection(list);

  // sub-list "Printing"
  Core::Utils::SectionSpecs printing{snox, "Printing"};

  {
    printing.specs.emplace_back(
        parameter<bool>("Error", {.description = "", .default_value = false}));
    printing.specs.emplace_back(
        parameter<bool>("Warning", {.description = "", .default_value = true}));
    printing.specs.emplace_back(
        parameter<bool>("Outer Iteration", {.description = "", .default_value = true}));
    printing.specs.emplace_back(
        parameter<bool>("Inner Iteration", {.description = "", .default_value = true}));
    printing.specs.emplace_back(
        parameter<bool>("Parameters", {.description = "", .default_value = false}));
    printing.specs.emplace_back(
        parameter<bool>("Details", {.description = "", .default_value = false}));
    printing.specs.emplace_back(
        parameter<bool>("Outer Iteration StatusTest", {.description = "", .default_value = true}));
    printing.specs.emplace_back(
        parameter<bool>("Linear Solver Details", {.description = "", .default_value = false}));
    printing.specs.emplace_back(
        parameter<bool>("Test Details", {.description = "", .default_value = false}));
    printing.specs.emplace_back(
        parameter<bool>("Debug", {.description = "", .default_value = false}));
  }
  printing.move_into_collection(list);

  // sub-list "Status Test"
  Core::Utils::SectionSpecs statusTest{snox, "Status Test"};

  {
    statusTest.specs.emplace_back(
        Core::IO::InputSpecBuilders::parameter<std::optional<std::filesystem::path>>("XML File",
            {.description = "Filename of XML file with configuration of nox status test"}));
  }
  statusTest.move_into_collection(list);

  // sub-list "Solver Options"
  Core::Utils::SectionSpecs solverOptions{snox, "Solver Options"};

  {
    Teuchos::Array<std::string> meritFct = Teuchos::tuple<std::string>("Sum of Squares");
    Core::Utils::string_to_integral_parameter<NOX::Nln::MeritFunction::MeritFctName>(
        "Merit Function", "Sum of Squares", "", meritFct,
        Teuchos::tuple<NOX::Nln::MeritFunction::MeritFctName>(
            NOX::Nln::MeritFunction::mrtfct_sum_of_squares),
        solverOptions);

    std::vector<std::string> status_test_check_type_valid_input = {"Complete", "Minimal", "None"};
    solverOptions.specs.emplace_back(deprecated_selection<std::string>("Status Test Check Type",
        status_test_check_type_valid_input, {.description = "", .default_value = "Complete"}));
  }
  solverOptions.move_into_collection(list);

  // sub-sub-sub-list "Linear Solver"
  Core::Utils::SectionSpecs linearSolver{newton, "Linear Solver"};

  {
    // convergence criteria adaptivity
    linearSolver.specs.emplace_back(parameter<bool>("Adaptive Control",
        {.description =
                "Switch on adaptive control of linear solver tolerance for nonlinear solution",
            .default_value = false}));
    linearSolver.specs.emplace_back(parameter<double>("Adaptive Control Objective",
        {.description = "The linear solver shall be this much better than the current nonlinear "
                        "residual in the nonlinear convergence limit",
            .default_value = 0.1}));
    linearSolver.specs.emplace_back(parameter<bool>("Zero Initial Guess",
        {.description = "Zero out the delta X vector if requested.", .default_value = true}));
    linearSolver.specs.emplace_back(parameter<bool>("Computing Scaling Manually",
        {.description =
                "Allows the manually scaling of your linear system (not supported at the moment).",
            .default_value = false}));
    linearSolver.specs.emplace_back(parameter<bool>("Output Solver Details",
        {.description = "Switch the linear solver output on and off.", .default_value = true}));
  }
  linearSolver.move_into_collection(list);
}

FOUR_C_NAMESPACE_CLOSE
