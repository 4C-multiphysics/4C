// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_inpar_porofluidmultiphase.hpp"

#include "4C_inpar_bio.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN

void Inpar::POROFLUIDMULTIPHASE::set_valid_parameters(
    std::map<std::string, Core::IO::InputSpec>& list)
{
  using Teuchos::tuple;
  using namespace Core::IO::InputSpecBuilders;

  Core::Utils::SectionSpecs porofluidmultiphasedyn{"POROFLUIDMULTIPHASE DYNAMIC"};

  porofluidmultiphasedyn.specs.emplace_back(parameter<double>(
      "MAXTIME", {.description = "Total simulation time", .default_value = 1000.0}));
  porofluidmultiphasedyn.specs.emplace_back(parameter<int>(
      "NUMSTEP", {.description = "Total number of time steps", .default_value = 20}));
  porofluidmultiphasedyn.specs.emplace_back(
      parameter<double>("TIMESTEP", {.description = "Time increment dt", .default_value = 0.1}));
  porofluidmultiphasedyn.specs.emplace_back(parameter<int>(
      "RESULTSEVERY", {.description = "Increment for writing solution", .default_value = 1}));
  porofluidmultiphasedyn.specs.emplace_back(parameter<int>(
      "RESTARTEVERY", {.description = "Increment for writing restart", .default_value = 1}));

  porofluidmultiphasedyn.specs.emplace_back(parameter<double>(
      "THETA", {.description = "One-step-theta time integration factor", .default_value = 0.5}));

  Core::Utils::string_to_integral_parameter<TimeIntegrationScheme>("TIMEINTEGR", "One_Step_Theta",
      "Time Integration Scheme", tuple<std::string>("One_Step_Theta"),
      tuple<TimeIntegrationScheme>(timeint_one_step_theta), porofluidmultiphasedyn);

  Core::Utils::string_to_integral_parameter<CalcError>("CALCERROR", "No",
      "compute error compared to analytical solution",
      tuple<std::string>("No", "error_by_function"),
      tuple<CalcError>(calcerror_no, calcerror_byfunction), porofluidmultiphasedyn);

  porofluidmultiphasedyn.specs.emplace_back(parameter<int>(
      "CALCERRORNO", {.description = "function number for porofluidmultiphase error computation",
                         .default_value = -1}));

  // linear solver id used for porofluidmultiphase problems
  porofluidmultiphasedyn.specs.emplace_back(parameter<int>("LINEAR_SOLVER",
      {.description = "number of linear solver used for the porofluidmultiphase problem",
          .default_value = -1}));
  porofluidmultiphasedyn.specs.emplace_back(parameter<int>(
      "ITEMAX", {.description = "max. number of nonlin. iterations", .default_value = 10}));
  porofluidmultiphasedyn.specs.emplace_back(parameter<double>("ABSTOLRES",
      {.description =
              "Absolute tolerance for deciding if residual of nonlinear problem is already zero",
          .default_value = 1e-14}));

  // convergence criteria adaptivity
  porofluidmultiphasedyn.specs.emplace_back(parameter<bool>("ADAPTCONV",
      {.description =
              "Switch on adaptive control of linear solver tolerance for nonlinear solution",
          .default_value = false}));
  porofluidmultiphasedyn.specs.emplace_back(parameter<double>("ADAPTCONV_BETTER",
      {.description = "The linear solver shall be this much better than the current nonlinear "
                      "residual in the nonlinear convergence limit",
          .default_value = 0.1}));

  // parameters for finite difference check
  Core::Utils::string_to_integral_parameter<FdCheck>("FDCHECK", "none",
      "flag for finite difference check: none, local, or global",
      tuple<std::string>("none",
          "global"),  // perform finite difference check on time integrator level
      tuple<FdCheck>(fdcheck_none, fdcheck_global), porofluidmultiphasedyn);
  porofluidmultiphasedyn.specs.emplace_back(parameter<double>(
      "FDCHECKEPS", {.description = "dof perturbation magnitude for finite difference check (1.e-6 "
                                    "seems to work very well, whereas smaller values don't)",
                        .default_value = 1.e-6}));
  porofluidmultiphasedyn.specs.emplace_back(parameter<double>("FDCHECKTOL",
      {.description = "relative tolerance for finite difference check", .default_value = 1.e-6}));
  porofluidmultiphasedyn.specs.emplace_back(parameter<bool>(
      "SKIPINITDER", {.description = "Flag to skip computation of initial time derivative",
                         .default_value = true}));
  porofluidmultiphasedyn.specs.emplace_back(parameter<bool>("OUTPUT_SATANDPRESS",
      {.description = "Flag if output of saturations and pressures should be calculated",
          .default_value = true}));
  porofluidmultiphasedyn.specs.emplace_back(parameter<bool>(
      "OUTPUT_SOLIDPRESS", {.description = "Flag if output of solid pressure should be calculated",
                               .default_value = true}));
  porofluidmultiphasedyn.specs.emplace_back(parameter<bool>("OUTPUT_POROSITY",
      {.description = "Flag if output of porosity should be calculated", .default_value = true}));
  porofluidmultiphasedyn.specs.emplace_back(parameter<bool>("OUTPUT_PHASE_VELOCITIES",
      {.description = "Flag if output of phase velocities should be calculated",
          .default_value = true}));

  // Biot stabilization
  porofluidmultiphasedyn.specs.emplace_back(parameter<bool>("STAB_BIOT",
      {.description = "Flag to (de)activate BIOT stabilization.", .default_value = false}));
  porofluidmultiphasedyn.specs.emplace_back(parameter<double>("STAB_BIOT_SCALING",
      {.description =
              "Scaling factor for stabilization parameter for biot stabilization of porous flow.",
          .default_value = 1.0}));

  Core::Utils::string_to_integral_parameter<VectorNorm>("VECTORNORM_RESF", "L2",
      "type of norm to be applied to residuals",
      tuple<std::string>("L1", "L1_Scaled", "L2", "Rms", "Inf"),
      tuple<VectorNorm>(Inpar::POROFLUIDMULTIPHASE::norm_l1,
          Inpar::POROFLUIDMULTIPHASE::norm_l1_scaled, Inpar::POROFLUIDMULTIPHASE::norm_l2,
          Inpar::POROFLUIDMULTIPHASE::norm_rms, Inpar::POROFLUIDMULTIPHASE::norm_inf),
      porofluidmultiphasedyn);

  Core::Utils::string_to_integral_parameter<VectorNorm>("VECTORNORM_INC", "L2",
      "type of norm to be applied to residuals",
      tuple<std::string>("L1", "L1_Scaled", "L2", "Rms", "Inf"),
      tuple<VectorNorm>(Inpar::POROFLUIDMULTIPHASE::norm_l1,
          Inpar::POROFLUIDMULTIPHASE::norm_l1_scaled, Inpar::POROFLUIDMULTIPHASE::norm_l2,
          Inpar::POROFLUIDMULTIPHASE::norm_rms, Inpar::POROFLUIDMULTIPHASE::norm_inf),
      porofluidmultiphasedyn);

  // Iterationparameters
  porofluidmultiphasedyn.specs.emplace_back(parameter<double>(
      "TOLRES", {.description = "tolerance in the residual norm for the Newton iteration",
                    .default_value = 1e-6}));
  porofluidmultiphasedyn.specs.emplace_back(parameter<double>(
      "TOLINC", {.description = "tolerance in the increment norm for the Newton iteration",
                    .default_value = 1e-6}));

  Core::Utils::string_to_integral_parameter<InitialField>("INITIALFIELD", "zero_field",
      "Initial Field for transport problem",
      tuple<std::string>("zero_field", "field_by_function", "field_by_condition"),
      tuple<InitialField>(
          initfield_zero_field, initfield_field_by_function, initfield_field_by_condition),
      porofluidmultiphasedyn);

  porofluidmultiphasedyn.specs.emplace_back(parameter<int>("INITFUNCNO",
      {.description = "function number for scalar transport initial field", .default_value = -1}));

  Core::Utils::string_to_integral_parameter<DivContAct>("DIVERCONT", "stop",
      "What to do with time integration when Newton-Raphson iteration failed",
      tuple<std::string>("stop", "continue"), tuple<DivContAct>(divcont_stop, divcont_continue),
      porofluidmultiphasedyn);

  porofluidmultiphasedyn.specs.emplace_back(parameter<int>("FLUX_PROJ_SOLVER",
      {.description = "Number of linear solver used for L2 projection", .default_value = -1}));

  Core::Utils::string_to_integral_parameter<FluxReconstructionMethod>("FLUX_PROJ_METHOD", "none",
      "Flag to (de)activate flux reconstruction.", tuple<std::string>("none", "L2_projection"),
      tuple<FluxReconstructionMethod>(
          gradreco_none,  // no convective streamline edge-based stabilization
          gradreco_l2     // pressure edge-based stabilization as ghost penalty around cut elements
          ),
      porofluidmultiphasedyn);

  // functions used for domain integrals
  porofluidmultiphasedyn.specs.emplace_back(parameter<std::string>("DOMAININT_FUNCT",
      {.description = "functions used for domain integrals", .default_value = "-1.0"}));

  // coupling with 1D artery network active
  porofluidmultiphasedyn.specs.emplace_back(parameter<bool>("ARTERY_COUPLING",
      {.description = "Coupling with 1D blood vessels.", .default_value = false}));

  porofluidmultiphasedyn.specs.emplace_back(parameter<double>("STARTING_DBC_TIME_END",
      {.description = "End time for the starting Dirichlet BC.", .default_value = -1.0}));

  porofluidmultiphasedyn.specs.emplace_back(parameter<std::string>("STARTING_DBC_ONOFF",
      {.description = "Switching the starting Dirichlet BC on or off.", .default_value = "0"}));

  porofluidmultiphasedyn.specs.emplace_back(parameter<std::string>("STARTING_DBC_FUNCT",
      {.description = "Function prescribing the starting Dirichlet BC.", .default_value = "0"}));

  porofluidmultiphasedyn.move_into_collection(list);

  // ----------------------------------------------------------------------
  // artery mesh tying
  Core::Utils::SectionSpecs porofluidmultiphasemshtdyn{porofluidmultiphasedyn, "ARTERY COUPLING"};

  // maximum number of segments per artery element for 1D-3D artery coupling
  porofluidmultiphasemshtdyn.specs.emplace_back(parameter<int>("MAXNUMSEGPERARTELE",
      {.description = "maximum number of segments per artery element for 1D-3D artery coupling",
          .default_value = 5}));

  // penalty parameter
  porofluidmultiphasemshtdyn.specs.emplace_back(parameter<double>("PENALTY",
      {.description = "Penalty parameter for line-based coupling", .default_value = 1000.0}));

  Core::Utils::string_to_integral_parameter<
      Inpar::ArteryNetwork::ArteryPoroMultiphaseScatraCouplingMethod>("ARTERY_COUPLING_METHOD",
      "None", "Coupling method for artery coupling.",
      tuple<std::string>("None", "Nodal", "GPTS", "MP", "NTP"),
      tuple<Inpar::ArteryNetwork::ArteryPoroMultiphaseScatraCouplingMethod>(
          Inpar::ArteryNetwork::ArteryPoroMultiphaseScatraCouplingMethod::none,   // none
          Inpar::ArteryNetwork::ArteryPoroMultiphaseScatraCouplingMethod::nodal,  // Nodal Coupling
          Inpar::ArteryNetwork::ArteryPoroMultiphaseScatraCouplingMethod::
              gpts,  // Gauss-Point-To-Segment
          Inpar::ArteryNetwork::ArteryPoroMultiphaseScatraCouplingMethod::mp,  // Mortar Penalty
          Inpar::ArteryNetwork::ArteryPoroMultiphaseScatraCouplingMethod::ntp  // 1Dnode-to-point in
                                                                               // 2D/3D
          ),
      porofluidmultiphasemshtdyn);

  // coupled artery dofs for mesh tying
  porofluidmultiphasemshtdyn.specs.emplace_back(parameter<std::string>("COUPLEDDOFS_ART",
      {.description = "coupled artery dofs for mesh tying", .default_value = "-1.0"}));

  // coupled porofluid dofs for mesh tying
  porofluidmultiphasemshtdyn.specs.emplace_back(parameter<std::string>("COUPLEDDOFS_PORO",
      {.description = "coupled porofluid dofs for mesh tying", .default_value = "-1.0"}));

  // functions for coupling (artery part)
  porofluidmultiphasemshtdyn.specs.emplace_back(parameter<std::string>("REACFUNCT_ART",
      {.description = "functions for coupling (artery part)", .default_value = "-1"}));

  // scale for coupling (artery part)
  porofluidmultiphasemshtdyn.specs.emplace_back(parameter<std::string>(
      "SCALEREAC_ART", {.description = "scale for coupling (artery part)", .default_value = "0"}));

  // functions for coupling (porofluid part)
  porofluidmultiphasemshtdyn.specs.emplace_back(parameter<std::string>("REACFUNCT_CONT",
      {.description = "functions for coupling (porofluid part)", .default_value = "-1"}));

  // scale for coupling (porofluid part)
  porofluidmultiphasemshtdyn.specs.emplace_back(parameter<std::string>("SCALEREAC_CONT",
      {.description = "scale for coupling (porofluid part)", .default_value = "0"}));

  // Flag if artery elements are evaluated in reference or current configuration
  porofluidmultiphasemshtdyn.specs.emplace_back(parameter<bool>("EVALUATE_IN_REF_CONFIG",
      {.description = "Flag if artery elements are evaluated in reference or current configuration",
          .default_value = true}));

  // Flag if 1D-3D coupling should be evaluated on lateral (cylinder) surface of embedded artery
  // elements
  porofluidmultiphasemshtdyn.specs.emplace_back(parameter<bool>("LATERAL_SURFACE_COUPLING",
      {.description = "Flag if 1D-3D coupling should be evaluated on lateral (cylinder) surface of "
                      "embedded artery elements",
          .default_value = false}));


  // Number of integration patches per 1D element in axial direction for lateral surface coupling
  porofluidmultiphasemshtdyn.specs.emplace_back(parameter<int>("NUMPATCH_AXI",
      {.description = "Number of integration patches per 1D element in axial direction for "
                      "lateral surface coupling",
          .default_value = 1}));

  // Number of integration patches per 1D element in radial direction for lateral surface coupling
  porofluidmultiphasemshtdyn.specs.emplace_back(parameter<int>("NUMPATCH_RAD",
      {.description = "Number of integration patches per 1D element in radial direction for "
                      "lateral surface coupling",
          .default_value = 1}));

  // Flag if blood vessel volume fraction should be output
  porofluidmultiphasemshtdyn.specs.emplace_back(parameter<bool>("OUTPUT_BLOODVESSELVOLFRAC",
      {.description = "Flag if output of blood vessel volume fraction should be calculated",
          .default_value = false}));

  // Flag if summary of coupling-pairs should be printed
  porofluidmultiphasemshtdyn.specs.emplace_back(parameter<bool>("PRINT_OUT_SUMMARY_PAIRS",
      {.description = "Flag if summary of coupling-pairs should be printed",
          .default_value = false}));

  // Flag if free-hanging elements (after blood vessel collapse) should be deleted
  porofluidmultiphasemshtdyn.specs.emplace_back(parameter<bool>("DELETE_FREE_HANGING_ELES",
      {.description =
              "Flag if free-hanging elements (after blood vessel collapse) should be deleted",
          .default_value = false}));

  // components whose size is smaller than this fraction of the total network size are also deleted
  porofluidmultiphasemshtdyn.specs.emplace_back(parameter<double>("DELETE_SMALL_FREE_HANGING_COMPS",
      {.description = "Small connected components whose size is smaller than this fraction of the "
                      "overall network size are additionally deleted (a valid choice of this "
                      "parameter should lie between 0 and 1)",
          .default_value = -1.0}));

  porofluidmultiphasemshtdyn.move_into_collection(list);
}

FOUR_C_NAMESPACE_CLOSE
