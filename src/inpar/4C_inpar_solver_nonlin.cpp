// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_inpar_solver_nonlin.hpp"

#include "4C_io_input_spec_builders.hpp"
#include "4C_solver_nonlin_nox_enum_lists.hpp"
FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Inpar::NlnSol::set_valid_parameters(std::map<std::string, Core::IO::InputSpec>& list)
{
  using namespace Core::IO::InputSpecBuilders;

  /*----------------------------------------------------------------------*
   * parameters for NOX - non-linear solution
   *----------------------------------------------------------------------*/
  list["STRUCT NOX"] = group("STRUCT NOX",
      {

          deprecated_selection<std::string>("Nonlinear Solver",
              {"Line Search Based", "Pseudo Transient", "Trust Region Based",
                  "Inexact Trust Region Based", "Tensor Based", "Single Step"},
              {.description = "Choose a nonlinear solver method.",
                  .default_value = "Line Search Based"})},
      {.defaultable = true});

  // sub-list direction
  list["STRUCT NOX/Direction"] = group("STRUCT NOX/Direction",
      {

          deprecated_selection<std::string>("Method",
              {"Newton", "Steepest Descent", "NonlinearCG", "Broyden", "User Defined"},
              {.description = "Choose a direction method for the nonlinear solver.",
                  .default_value = "Newton"}),


          deprecated_selection<std::string>("User Defined Method", {"Newton", "Modified Newton"},
              {.description = "Choose a user-defined direction method.",
                  .default_value = "Modified Newton"})},
      {.defaultable = true});

  // sub-sub-list "Newton"
  list["STRUCT NOX/Direction/Newton"] = group("STRUCT NOX/Direction/Newton",
      {

          deprecated_selection<std::string>("Forcing Term Method", {"Constant", "Type 1", "Type 2"},
              {.description = "", .default_value = "Constant"}),

          parameter<double>("Forcing Term Initial Tolerance",
              {.description = "initial linear solver tolerance", .default_value = 0.1}),
          parameter<double>(
              "Forcing Term Minimum Tolerance", {.description = "", .default_value = 1.0e-6}),
          parameter<double>(
              "Forcing Term Maximum Tolerance", {.description = "", .default_value = 0.01}),
          parameter<double>("Forcing Term Alpha",
              {.description = "used only by \"Type 2\"", .default_value = 1.5}),
          parameter<double>("Forcing Term Gamma",
              {.description = "used only by \"Type 2\"", .default_value = 0.9}),
          parameter<bool>("Rescue Bad Newton Solve",
              {.description =
                      "If set to true, we will use the computed direction even if the linear "
                      "solve does not achieve the tolerance specified by the forcing term",
                  .default_value = true})},
      {.defaultable = true});

  // sub-sub-list "Steepest Descent"
  list["STRUCT NOX/Direction/Steepest Descent"] = group("STRUCT NOX/Direction/Steepest Descent",
      {

          deprecated_selection<std::string>("Scaling Type",
              {"2-Norm", "Quadratic Model Min", "F 2-Norm", "None"},
              {.description = "", .default_value = "None"})},
      {.defaultable = true});

  // sub-list "Pseudo Transient"
  list["STRUCT NOX/Pseudo Transient"] = group("STRUCT NOX/Pseudo Transient",
      {

          parameter<double>("deltaInit",
              {.description = "Initial time step size. If its negative, the initial time "
                              "step is calculated automatically.",
                  .default_value = -1.0}),
          parameter<double>("deltaMax",
              {.description =
                      "Maximum time step size. If the new step size is greater than this value, "
                      "the "
                      "transient terms will be eliminated from the Newton iteration resulting in a "
                      "full "
                      "Newton solve.",
                  .default_value = std::numeric_limits<double>::max()}),
          parameter<double>(
              "deltaMin", {.description = "Minimum step size.", .default_value = 1.0e-5}),
          parameter<int>("Max Number of PTC Iterations",
              {.description = "", .default_value = std::numeric_limits<int>::max()}),

          parameter<double>("SER_alpha", {.description = "Exponent of SET.", .default_value = 1.0}),
          parameter<double>("ScalingFactor",
              {.description = "Scaling Factor for ptc matrix.", .default_value = 1.0}),

          deprecated_selection<std::string>("Time Step Control",
              {"SER", "Switched Evolution Relaxation", "TTE", "Temporal Truncation Error", "MRR",
                  "Model Reduction Ratio"},
              {.description = "", .default_value = "SER"}),


          deprecated_selection<std::string>("Norm Type for TSC",
              {"Two Norm", "One Norm", "Max Norm"},
              {.description = "Norm Type for the time step control", .default_value = "Max Norm"}),

          deprecated_selection<std::string>("Scaling Type",
              {"Identity", "CFL Diagonal", "Lumped Mass", "Element based"},
              {.description = "Type of the scaling matrix for the PTC method.",
                  .default_value = "Identity"}),


          deprecated_selection<std::string>("Build scaling operator",
              {"every iter", "every timestep"},
              {.description = "Build scaling operator in every iteration or timestep",
                  .default_value = "every timestep"})},
      {.defaultable = true});

  // sub-list "Line Search"
  list["STRUCT NOX/Line Search"] = group("STRUCT NOX/Line Search",
      {

          deprecated_selection<std::string>("Method",
              {"Full Step", "Backtrack", "Polynomial", "More'-Thuente", "User Defined"},
              {.description = "", .default_value = "Full Step"}),



          parameter<::NOX::StatusTest::CheckType>("Inner Status Test Check Type",
              {.description = "Specify the check type for the inner status tests.",
                  .default_value = ::NOX::StatusTest::Minimal})},
      {.defaultable = true});

  // sub-sub-list "Full Step"
  list["STRUCT NOX/Line Search/Full Step"] = group("STRUCT NOX/Line Search/Full Step",
      {

          parameter<double>(
              "Full Step", {.description = "length of a full step", .default_value = 1.0})},
      {.defaultable = true});

  // sub-sub-list "Backtrack"
  list["STRUCT NOX/Line Search/Backtrack"] = group("STRUCT NOX/Line Search/Backtrack",
      {

          parameter<double>(
              "Default Step", {.description = "starting step length", .default_value = 1.0}),
          parameter<double>("Minimum Step",
              {.description = "minimum acceptable step length", .default_value = 1.0e-12}),
          parameter<double>(
              "Recovery Step", {.description = "step to take when the line search fails "
                                               "(defaults to value for \"Default Step\")",
                                   .default_value = 1.0}),
          parameter<int>(
              "Max Iters", {.description = "maximum number of iterations", .default_value = 50}),
          parameter<double>("Reduction Factor",
              {.description = "A multiplier between zero and one that reduces the "
                              "step size between line search iterations",
                  .default_value = 0.5}),
          parameter<bool>("Allow Exceptions",
              {.description =
                      "Set to true, if exceptions during the force evaluation and backtracking "
                      "routine should be allowed.",
                  .default_value = false})},
      {.defaultable = true});

  // sub-sub-list "Polynomial"
  list["STRUCT NOX/Line Search/Polynomial"] = group("STRUCT NOX/Line Search/Polynomial",
      {

          parameter<double>(
              "Default Step", {.description = "Starting step length", .default_value = 1.0}),
          parameter<int>(
              "Max Iters", {.description = "Maximum number of line search iterations. The search "
                                           "fails if the number of iterations exceeds this value",
                               .default_value = 100}),
          parameter<double>("Minimum Step",
              {.description = "Minimum acceptable step length. The search fails if the "
                              "computed $\\lambda_k$ is less than this value",
                  .default_value = 1.0e-12}),


          deprecated_selection<std::string>("Recovery Step Type",
              {"Constant", "Last Computed Step"},
              {.description = "Determines the step size to take when the line search fails",
                  .default_value = "Constant"}),

          parameter<double>(
              "Recovery Step", {.description = "The value of the step to take when the line search "
                                               "fails. Only used if the \"Recovery "
                                               "Step Type\" is set to \"Constant\"",
                                   .default_value = 1.0}),


          deprecated_selection<std::string>("Interpolation Type",
              {"Quadratic", "Quadratic3", "Cubic"},
              {.description = "Type of interpolation that should be used",
                  .default_value = "Cubic"}),

          parameter<double>("Min Bounds Factor",
              {.description =
                      "Choice for $\\gamma_{\\min}$, i.e., the factor that limits the minimum "
                      "size of the new step based on the previous step",
                  .default_value = 0.1}),
          parameter<double>("Max Bounds Factor",
              {.description =
                      "Choice for $\\gamma_{\\max}$, i.e., the factor that limits the maximum "
                      "size of the new step based on the previous step",
                  .default_value = 0.5}),

          deprecated_selection<std::string>("Sufficient Decrease Condition",
              {"Armijo-Goldstein", "Ared/Pred", "None"},
              {.description = "Choice to use for the sufficient decrease condition",
                  .default_value = "Armijo-Goldstein"}),

          parameter<double>(
              "Alpha Factor", {.description = "Parameter choice for sufficient decrease condition",
                                  .default_value = 1.0e-4}),
          parameter<bool>("Force Interpolation",
              {.description = "Set to true if at least one interpolation step should be used. The "
                              "default is false which means that the line search will stop if the "
                              "default step length satisfies the convergence criteria",
                  .default_value = false}),
          parameter<bool>("Use Counters",
              {.description =
                      "Set to true if we should use counters and then output the result to the "
                      "parameter list as described in Output Parameters",
                  .default_value = true}),
          parameter<int>("Maximum Iteration for Increase",
              {.description = "Maximum index of the nonlinear iteration for which we allow a "
                              "relative increase",
                  .default_value = 0}),

          parameter<double>(
              "Allowed Relative Increase", {.description = "", .default_value = 100.0})},
      {.defaultable = true});

  // sub-sub-list "More'-Thuente"
  list["STRUCT NOX/Line Search/More'-Thuente"] = group("STRUCT NOX/Line Search/More'-Thuente",
      {

          parameter<double>("Sufficient Decrease",
              {.description = "The ftol in the sufficient decrease condition",
                  .default_value = 1.0e-4}),
          parameter<double>("Curvature Condition",
              {.description = "The gtol in the curvature condition", .default_value = 0.9999}),
          parameter<double>(
              "Interval Width", {.description = "The maximum width of the interval containing the "
                                                "minimum of the modified function",
                                    .default_value = 1.0e-15}),
          parameter<double>("Maximum Step",
              {.description = "maximum allowable step length", .default_value = 1.0e6}),
          parameter<double>("Minimum Step",
              {.description = "minimum allowable step length", .default_value = 1.0e-12}),
          parameter<int>("Max Iters",
              {.description =
                      "maximum number of right-hand-side and corresponding Jacobian evaluations",
                  .default_value = 20}),
          parameter<double>(
              "Default Step", {.description = "starting step length", .default_value = 1.0}),


          deprecated_selection<std::string>("Recovery Step Type",
              {"Constant", "Last Computed Step"},
              {.description = "Determines the step size to take when the line search fails",
                  .default_value = "Constant"}),

          parameter<double>(
              "Recovery Step", {.description = "The value of the step to take when the line search "
                                               "fails. Only used if the \"Recovery "
                                               "Step Type\" is set to \"Constant\"",
                                   .default_value = 1.0}),

          deprecated_selection<std::string>("Sufficient Decrease Condition",
              {"Armijo-Goldstein", "Ared/Pred", "None"},
              {.description = "Choice to use for the sufficient decrease condition",
                  .default_value = "Armijo-Goldstein"}),

          parameter<bool>("Optimize Slope Calculation",
              {.description =
                      "Boolean value. If set to true the value of $s^T J^T F$ is estimated using "
                      "a directional derivative in a call to "
                      "::NOX::LineSearch::Common::computeSlopeWithOutJac. If false the slope "
                      "computation is computed with the ::NOX::LineSearch::Common::computeSlope "
                      "method. Setting this to true eliminates having to compute the Jacobian at "
                      "each inner iteration of the More'-Thuente line search",
                  .default_value = false})},
      {.defaultable = true});

  // sub-list "Trust Region"
  list["STRUCT NOX/Trust Region"] = group("STRUCT NOX/Trust Region",
      {

          parameter<double>("Minimum Trust Region Radius",
              {.description = "Minimum allowable trust region radius", .default_value = 1.0e-6}),
          parameter<double>("Maximum Trust Region Radius",
              {.description = "Maximum allowable trust region radius", .default_value = 1.0e+9}),
          parameter<double>("Minimum Improvement Ratio",
              {.description = "Minimum improvement ratio to accept the step",
                  .default_value = 1.0e-4}),
          parameter<double>("Contraction Trigger Ratio",
              {.description =
                      "If the improvement ratio is less than this value, then the trust region is "
                      "contracted by "
                      "the amount specified by the \"Contraction Factor\". Must be larger than "
                      "\"Minimum "
                      "Improvement Ratio\"",
                  .default_value = 0.1}),

          parameter<double>("Contraction Factor", {.description = "", .default_value = 0.25}),
          parameter<double>("Expansion Trigger Ratio",
              {.description = "If the improvement ratio is greater than this value, then the trust "
                              "region is contracted "
                              "by the amount specified by the \"Expansion Factor\"",
                  .default_value = 0.75}),

          parameter<double>("Expansion Factor", {.description = "", .default_value = 4.0}),

          parameter<double>("Recovery Step", {.description = "", .default_value = 1.0})},
      {.defaultable = true});

  // sub-list "Printing"
  list["STRUCT NOX/Printing"] = group("STRUCT NOX/Printing",
      {

          parameter<bool>("Error", {.description = "", .default_value = false}),

          parameter<bool>("Warning", {.description = "", .default_value = true}),

          parameter<bool>("Outer Iteration", {.description = "", .default_value = true}),

          parameter<bool>("Inner Iteration", {.description = "", .default_value = true}),

          parameter<bool>("Parameters", {.description = "", .default_value = false}),

          parameter<bool>("Details", {.description = "", .default_value = false}),

          parameter<bool>("Outer Iteration StatusTest", {.description = "", .default_value = true}),

          parameter<bool>("Linear Solver Details", {.description = "", .default_value = false}),

          parameter<bool>("Test Details", {.description = "", .default_value = false}),

          parameter<bool>("Debug", {.description = "", .default_value = false})},
      {.defaultable = true});

  // sub-list "Status Test"
  list["STRUCT NOX/Status Test"] = group("STRUCT NOX/Status Test",
      {

          Core::IO::InputSpecBuilders::parameter<std::optional<std::filesystem::path>>("XML File",
              {.description = "Filename of XML file with configuration of nox status test"})},
      {.defaultable = true});

  // sub-list "Solver Options"
  list["STRUCT NOX/Solver Options"] = group("STRUCT NOX/Solver Options",
      {

          deprecated_selection<NOX::Nln::MeritFunction::MeritFctName>("Merit Function",
              {
                  {"Sum of Squares", NOX::Nln::MeritFunction::mrtfct_sum_of_squares},
              },
              {.description = "", .default_value = NOX::Nln::MeritFunction::mrtfct_sum_of_squares}),

          deprecated_selection<std::string>("Status Test Check Type",
              {"Complete", "Minimal", "None"}, {.description = "", .default_value = "Complete"})},
      {.defaultable = true});

  // sub-sub-sub-list "Linear Solver"
  list["STRUCT NOX/Direction/Newton/Linear Solver"] = group(
      "STRUCT NOX/Direction/Newton/Linear Solver",
      {

          // convergence criteria adaptivity
          parameter<bool>(
              "Adaptive Control", {.description = "Switch on adaptive control of linear solver "
                                                  "tolerance for nonlinear solution",
                                      .default_value = false}),
          parameter<double>("Adaptive Control Objective",
              {.description =
                      "The linear solver shall be this much better than the current nonlinear "
                      "residual in the nonlinear convergence limit",
                  .default_value = 0.1}),
          parameter<bool>("Zero Initial Guess",
              {.description = "Zero out the delta X vector if requested.", .default_value = true}),
          parameter<bool>("Computing Scaling Manually",
              {.description =
                      "Allows the manually scaling of your linear system (not supported at the "
                      "moment).",
                  .default_value = false}),
          parameter<bool>("Output Solver Details",
              {.description = "Switch the linear solver output on and off.",
                  .default_value = true})},
      {.defaultable = true});
}

FOUR_C_NAMESPACE_CLOSE