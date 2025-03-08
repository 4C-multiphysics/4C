// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_inpar_ehl.hpp"

#include "4C_fem_condition_definition.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN



void Inpar::EHL::set_valid_parameters(std::map<std::string, Core::IO::InputSpec>& list)
{
  using Teuchos::tuple;
  using namespace Core::IO::InputSpecBuilders;

  Core::Utils::SectionSpecs ehldyn{"ELASTO HYDRO DYNAMIC"};

  // Output type
  ehldyn.specs.emplace_back(parameter<double>("RESTARTEVERYTIME",
      {.description = "write restart possibility every RESTARTEVERY steps", .default_value = 0.0}));
  Core::Utils::int_parameter(
      "RESTARTEVERY", 1, "write restart possibility every RESTARTEVERY steps", ehldyn);
  // Time loop control
  Core::Utils::int_parameter("NUMSTEP", 200, "maximum number of Timesteps", ehldyn);
  ehldyn.specs.emplace_back(parameter<double>(
      "MAXTIME", {.description = "total simulation time", .default_value = 1000.0}));
  ehldyn.specs.emplace_back(
      parameter<double>("TIMESTEP", {.description = "time step size dt", .default_value = -1.0}));
  ehldyn.specs.emplace_back(parameter<bool>(
      "DIFFTIMESTEPSIZE", {.description = "use different step size for lubrication and solid",
                              .default_value = false}));
  ehldyn.specs.emplace_back(parameter<double>(
      "RESULTSEVERYTIME", {.description = "increment for writing solution", .default_value = 0.0}));
  Core::Utils::int_parameter("RESULTSEVERY", 1, "increment for writing solution", ehldyn);
  Core::Utils::int_parameter("ITEMAX", 10, "maximum number of iterations over fields", ehldyn);
  Core::Utils::int_parameter("ITEMIN", 1, "minimal number of iterations over fields", ehldyn);

  // Type of coupling strategy between the two fields
  Core::Utils::string_to_integral_parameter<FieldCoupling>("FIELDCOUPLING", "none",
      "Type of coupling strategy between fields", tuple<std::string>("none", "matching"),
      tuple<FieldCoupling>(coupling_none, coupling_matching), ehldyn);

  // Coupling strategy for EHL solvers
  Core::Utils::string_to_integral_parameter<SolutionSchemeOverFields>("COUPALGO", "ehl_Monolithic",
      "Coupling strategies for EHL solvers", tuple<std::string>("ehl_IterStagg", "ehl_Monolithic"),
      tuple<SolutionSchemeOverFields>(ehl_IterStagg, ehl_Monolithic), ehldyn);

  // set unprojectable nodes to zero pressure via Dirichlet condition
  ehldyn.specs.emplace_back(parameter<bool>("UNPROJ_ZERO_DBC",
      {.description = "set unprojectable nodes to zero pressure via Dirichlet condition",
          .default_value = false}));

  // use dry contact model
  ehldyn.specs.emplace_back(parameter<bool>("DRY_CONTACT_MODEL",
      {.description = "set unprojectable nodes to zero pressure via Dirichlet condition",
          .default_value = false}));


  ehldyn.move_into_collection(list);

  /*----------------------------------------------------------------------*/
  /* parameters for monolithic EHL */
  Core::Utils::SectionSpecs ehldynmono{ehldyn, "MONOLITHIC"};

  // convergence tolerance of EHL residual
  ehldynmono.specs.emplace_back(parameter<double>(
      "CONVTOL", {.description = "tolerance for convergence check of EHL", .default_value = 1e-6}));
  // Iterationparameters
  ehldynmono.specs.emplace_back(parameter<double>("TOLINC",
      {.description = "tolerance for convergence check of EHL-increment in monolithic EHL",
          .default_value = 1.0e-6}));

  Core::Utils::string_to_integral_parameter<ConvNorm>("NORM_RESF", "Abs",
      "type of norm for residual convergence check", tuple<std::string>("Abs", "Rel", "Mix"),
      tuple<ConvNorm>(convnorm_abs, convnorm_rel, convnorm_mix), ehldynmono);

  Core::Utils::string_to_integral_parameter<ConvNorm>("NORM_INC", "Abs",
      "type of norm for convergence check of primary variables in EHL",
      tuple<std::string>("Abs", "Rel", "Mix"),
      tuple<ConvNorm>(convnorm_abs, convnorm_rel, convnorm_mix), ehldynmono);


  Core::Utils::string_to_integral_parameter<BinaryOp>("NORMCOMBI_RESFINC", "Coupl_And_Single",
      "binary operator to combine primary variables and residual force values",
      tuple<std::string>(
          "And", "Or", "Coupl_Or_Single", "Coupl_And_Single", "And_Single", "Or_Single"),
      tuple<BinaryOp>(bop_and, bop_or, bop_coupl_or_single, bop_coupl_and_single, bop_and_single,
          bop_or_single),
      ehldynmono);

  Core::Utils::string_to_integral_parameter<VectorNorm>("ITERNORM", "Rms",
      "type of norm to be applied to residuals",
      tuple<std::string>("L1", "L1_Scaled", "L2", "Rms", "Inf"),
      tuple<VectorNorm>(norm_l1, norm_l1_scaled, norm_l2, norm_rms, norm_inf), ehldynmono);

  ehldynmono.specs.emplace_back(
      parameter<double>("PTCDT", {.description = "pseudo time step for pseudo-transient "
                                                 "continuation (PTC) stabilised Newton procedure",
                                     .default_value = 0.1}));

  // number of linear solver used for monolithic EHL
  Core::Utils::int_parameter(
      "LINEAR_SOLVER", -1, "number of linear solver used for monolithic EHL problems", ehldynmono);

  // convergence criteria adaptivity of monolithic EHL solver
  ehldynmono.specs.emplace_back(parameter<bool>("ADAPTCONV",
      {.description =
              "Switch on adaptive control of linear solver tolerance for nonlinear solution",
          .default_value = false}));
  ehldynmono.specs.emplace_back(parameter<double>("ADAPTCONV_BETTER",
      {.description = "The linear solver shall be this much better than the current nonlinear "
                      "residual in the nonlinear convergence limit",
          .default_value = 0.1}));

  ehldynmono.specs.emplace_back(parameter<bool>("INFNORMSCALING",
      {.description = "Scale blocks of matrix with row infnorm?", .default_value = true}));

  ehldynmono.move_into_collection(list);

  /*----------------------------------------------------------------------*/
  /* parameters for partitioned EHL */
  /*----------------------------------------------------------------------*/
  Core::Utils::SectionSpecs ehldynpart{ehldyn, "PARTITIONED"};

  // Solver parameter for relaxation of iterative staggered partitioned EHL
  ehldynpart.specs.emplace_back(parameter<double>("MAXOMEGA",
      {.description = "largest omega allowed for Aitken relaxation", .default_value = 10.0}));
  ehldynpart.specs.emplace_back(parameter<double>("MINOMEGA",
      {.description = "smallest omega allowed for Aitken relaxation", .default_value = 0.1}));
  ehldynpart.specs.emplace_back(parameter<double>(
      "STARTOMEGA", {.description = "fixed relaxation parameter", .default_value = 1.0}));

  // convergence tolerance of outer iteration loop
  ehldynpart.specs.emplace_back(parameter<double>("CONVTOL",
      {.description = "tolerance for convergence check of outer iteration within partitioned EHL",
          .default_value = 1e-6}));

  ehldynpart.move_into_collection(list);
}


void Inpar::EHL::set_valid_conditions(std::vector<Core::Conditions::ConditionDefinition>& condlist)
{
  using namespace Core::IO::InputSpecBuilders;
  /*--------------------------------------------------------------------*/
  // ehl mortar coupling

  Core::Conditions::ConditionDefinition lineehl("DESIGN LINE EHL MORTAR COUPLING CONDITIONS 2D",
      "EHLCoupling", "Line EHL Coupling", Core::Conditions::EHLCoupling, true,
      Core::Conditions::geometry_type_line);
  Core::Conditions::ConditionDefinition surfehl("DESIGN SURF EHL MORTAR COUPLING CONDITIONS 3D",
      "EHLCoupling", "Surface EHL Coupling", Core::Conditions::EHLCoupling, true,
      Core::Conditions::geometry_type_surface);

  const auto make_ehl_cond = [&condlist](Core::Conditions::ConditionDefinition& cond)
  {
    cond.add_component(parameter<int>("InterfaceID"));
    cond.add_component(deprecated_selection<std::string>(
        "Side", {"Master", "Slave"}, {.description = "interface side"}));
    cond.add_component(deprecated_selection<std::string>("Initialization", {"Inactive", "Active"},
        {.description = "initialization", .default_value = "Active"}));
    cond.add_component(
        parameter<double>("FrCoeffOrBound", {.description = "", .default_value = 0.0}));

    condlist.push_back(cond);
  };

  make_ehl_cond(lineehl);
  make_ehl_cond(surfehl);
}

FOUR_C_NAMESPACE_CLOSE
