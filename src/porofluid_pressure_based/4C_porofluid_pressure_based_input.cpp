// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_porofluid_pressure_based_input.hpp"

#include "4C_inpar_bio.hpp"
#include "4C_io_input_spec_builders.hpp"

#include <Teuchos_Time.hpp>

FOUR_C_NAMESPACE_OPEN

void PoroPressureBased::set_valid_parameters_porofluid(
    std::map<std::string, Core::IO::InputSpec>& list)
{
  using namespace Core::IO::InputSpecBuilders;

  list["POROFLUIDMULTIPHASE DYNAMIC"] = group("POROFLUIDMULTIPHASE DYNAMIC",
      {

          parameter<double>(
              "MAXTIME", {.description = "Total simulation time", .default_value = 1000.0}),
          parameter<int>(
              "NUMSTEP", {.description = "Total number of time steps", .default_value = 20}),

          parameter<double>("TIMESTEP", {.description = "Time increment dt", .default_value = 0.1}),
          parameter<int>("RESULTSEVERY",
              {.description = "Increment for writing solution", .default_value = 1}),
          parameter<int>(
              "RESTARTEVERY", {.description = "Increment for writing restart", .default_value = 1}),

          parameter<double>("THETA",
              {.description = "One-step-theta time integration factor", .default_value = 0.5}),


          deprecated_selection<TimeIntegrationScheme>("TIMEINTEGR",
              {
                  {"One_Step_Theta", TimeIntegrationScheme::one_step_theta},
              },
              {.description = "Time Integration Scheme",
                  .default_value = TimeIntegrationScheme::one_step_theta}),

          deprecated_selection<CalcError>("CALCERROR",
              {
                  {"No", CalcError::no},
                  {"error_by_function", CalcError::by_function},
              },
              {.description = "compute error compared to analytical solution",
                  .default_value = CalcError::no}),

          parameter<int>("CALCERRORNO",
              {.description = "function number for porofluidmultiphase error computation",
                  .default_value = -1}),

          // linear solver id used for porofluidmultiphase problems
          parameter<int>("LINEAR_SOLVER",
              {.description = "number of linear solver used for the porofluidmultiphase problem",
                  .default_value = -1}),
          parameter<int>(
              "ITEMAX", {.description = "max. number of nonlin. iterations", .default_value = 10}),
          parameter<double>(
              "ABSTOLRES", {.description = "Absolute tolerance for deciding if residual "
                                           "of nonlinear problem is already zero",
                               .default_value = 1e-14}),

          // convergence criteria adaptivity
          parameter<bool>("ADAPTCONV", {.description = "Switch on adaptive control of linear "
                                                       "solver tolerance for nonlinear solution",
                                           .default_value = false}),
          parameter<double>("ADAPTCONV_BETTER",
              {.description =
                      "The linear solver shall be this much better than the current nonlinear "
                      "residual in the nonlinear convergence limit",
                  .default_value = 0.1}),

          // parameters for finite difference check
          parameter<FdCheck>(
              "FDCHECK", {.description = "flag for finite difference check: none, local, or global",
                             .default_value = FdCheck::none}),
          parameter<double>("FDCHECKEPS",
              {.description = "dof perturbation magnitude for finite difference check (1.e-6 "
                              "seems to work very well, whereas smaller values don't)",
                  .default_value = 1.e-6}),
          parameter<double>(
              "FDCHECKTOL", {.description = "relative tolerance for finite difference check",
                                .default_value = 1.e-6}),
          parameter<bool>(
              "SKIPINITDER", {.description = "Flag to skip computation of initial time derivative",
                                 .default_value = true}),
          parameter<bool>("OUTPUT_SATANDPRESS",
              {.description = "Flag if output of saturations and pressures should be calculated",
                  .default_value = true}),
          parameter<bool>("OUTPUT_SOLIDPRESS",
              {.description = "Flag if output of solid pressure should be calculated",
                  .default_value = true}),
          parameter<bool>(
              "OUTPUT_POROSITY", {.description = "Flag if output of porosity should be calculated",
                                     .default_value = true}),
          parameter<bool>("OUTPUT_PHASE_VELOCITIES",
              {.description = "Flag if output of phase velocities should be calculated",
                  .default_value = true}),

          // Biot stabilization
          parameter<bool>("STAB_BIOT",
              {.description = "Flag to (de)activate BIOT stabilization.", .default_value = false}),
          parameter<double>(
              "STAB_BIOT_SCALING", {.description = "Scaling factor for stabilization parameter for "
                                                   "biot stabilization of porous flow.",
                                       .default_value = 1.0}),

          deprecated_selection<VectorNorm>("VECTORNORM_RESF",
              {
                  {"L1", VectorNorm::l1},
                  {"L1_Scaled", VectorNorm::l1_scaled},
                  {"L2", VectorNorm::l2},
                  {"Rms", VectorNorm::rms},
                  {"Inf", VectorNorm::inf},
              },
              {.description = "type of norm to be applied to residuals",
                  .default_value = VectorNorm::l2}),

          deprecated_selection<VectorNorm>("VECTORNORM_INC",
              {
                  {"L1", VectorNorm::l1},
                  {"L1_Scaled", VectorNorm::l1_scaled},
                  {"L2", VectorNorm::l2},
                  {"Rms", VectorNorm::rms},
                  {"Inf", VectorNorm::inf},
              },
              {.description = "type of norm to be applied to residuals",
                  .default_value = VectorNorm::l2}),

          // Iterationparameters
          parameter<double>(
              "TOLRES", {.description = "tolerance in the residual norm for the Newton iteration",
                            .default_value = 1e-6}),
          parameter<double>(
              "TOLINC", {.description = "tolerance in the increment norm for the Newton iteration",
                            .default_value = 1e-6}),

          deprecated_selection<InitialField>("INITIALFIELD",
              {
                  {"zero_field", InitialField::zero},
                  {"field_by_function", InitialField::by_function},
                  {"field_by_condition", InitialField::by_condition},
              },
              {.description = "Initial Field for the porofluid problem",
                  .default_value = InitialField::zero}),

          parameter<int>(
              "INITFUNCNO", {.description = "function number for scalar transport initial field",
                                .default_value = -1}),

          deprecated_selection<DivergenceAction>("DIVERCONT",
              {
                  {"stop", DivergenceAction::stop},
                  {"continue", DivergenceAction::continue_anyway},
              },
              {.description =
                      "What to do with time integration when Newton-Raphson iteration failed",
                  .default_value = DivergenceAction::stop}),

          parameter<int>(
              "FLUX_PROJ_SOLVER", {.description = "Number of linear solver used for L2 projection",
                                      .default_value = -1}),


          deprecated_selection<FluxReconstructionMethod>("FLUX_PROJ_METHOD",
              {
                  {"none", FluxReconstructionMethod::none},
                  {"L2_projection", FluxReconstructionMethod::l2},
              },
              {.description = "Flag to (de)activate flux reconstruction.",
                  .default_value = FluxReconstructionMethod::none}),

          // functions used for domain integrals
          parameter<std::string>("DOMAININT_FUNCT",
              {.description = "functions used for domain integrals", .default_value = "-1.0"}),

          // coupling with 1D artery network active
          parameter<bool>("ARTERY_COUPLING",
              {.description = "Coupling with 1D blood vessels.", .default_value = false}),

          parameter<double>("STARTING_DBC_TIME_END",
              {.description = "End time for the starting Dirichlet BC.", .default_value = -1.0}),

          parameter<std::string>("STARTING_DBC_ONOFF",
              {.description = "Switching the starting Dirichlet BC on or off.",
                  .default_value = "0"}),

          parameter<std::string>("STARTING_DBC_FUNCT",
              {.description = "Function prescribing the starting Dirichlet BC.",
                  .default_value = "0"})},
      {.defaultable =
              true});  // ----------------------------------------------------------------------
  // artery mesh tying
  list["POROFLUIDMULTIPHASE DYNAMIC/ARTERY COUPLING"] = group(
      "POROFLUIDMULTIPHASE DYNAMIC/ARTERY COUPLING",
      {

          // maximum number of segments per artery element for 1D-3D artery coupling
          parameter<int>("MAXNUMSEGPERARTELE",
              {.description =
                      "maximum number of segments per artery element for 1D-3D artery coupling",
                  .default_value = 5}),

          // penalty parameter
          parameter<double>("PENALTY", {.description = "Penalty parameter for line-based coupling",
                                           .default_value = 1000.0}),


          deprecated_selection<Inpar::ArteryNetwork::ArteryPorofluidElastScatraCouplingMethod>(
              "ARTERY_COUPLING_METHOD",
              {
                  {"None", Inpar::ArteryNetwork::ArteryPorofluidElastScatraCouplingMethod::none},
                  {"Nodal", Inpar::ArteryNetwork::ArteryPorofluidElastScatraCouplingMethod::nodal},
                  {"GPTS", Inpar::ArteryNetwork::ArteryPorofluidElastScatraCouplingMethod::gpts},
                  {"MP", Inpar::ArteryNetwork::ArteryPorofluidElastScatraCouplingMethod::mp},
                  {"NTP", Inpar::ArteryNetwork::ArteryPorofluidElastScatraCouplingMethod::ntp},
              },
              {.description = "Coupling method for artery coupling.",
                  .default_value =
                      Inpar::ArteryNetwork::ArteryPorofluidElastScatraCouplingMethod::none}),

          // coupled artery dofs for mesh tying
          parameter<std::string>("COUPLEDDOFS_ART",
              {.description = "coupled artery dofs for mesh tying", .default_value = "-1.0"}),

          // coupled porofluid dofs for mesh tying
          parameter<std::string>("COUPLEDDOFS_PORO",
              {.description = "coupled porofluid dofs for mesh tying", .default_value = "-1.0"}),

          // functions for coupling (artery part)
          parameter<std::string>("REACFUNCT_ART",
              {.description = "functions for coupling (artery part)", .default_value = "-1"}),

          // scale for coupling (artery part)
          parameter<std::string>("SCALEREAC_ART",
              {.description = "scale for coupling (artery part)", .default_value = "0"}),

          // functions for coupling (porofluid part)
          parameter<std::string>("REACFUNCT_CONT",
              {.description = "functions for coupling (porofluid part)", .default_value = "-1"}),

          // scale for coupling (porofluid part)
          parameter<std::string>("SCALEREAC_CONT",
              {.description = "scale for coupling (porofluid part)", .default_value = "0"}),

          // Flag if artery elements are evaluated in reference or current configuration
          parameter<bool>("EVALUATE_IN_REF_CONFIG",
              {.description =
                      "Flag if artery elements are evaluated in reference or current configuration",
                  .default_value = true}),

          // Flag if 1D-3D coupling should be evaluated on lateral (cylinder) surface of embedded
          // artery
          // elements
          parameter<bool>("LATERAL_SURFACE_COUPLING",
              {.description =
                      "Flag if 1D-3D coupling should be evaluated on lateral (cylinder) surface of "
                      "embedded artery elements",
                  .default_value = false}),


          // Number of integration patches per 1D element in axial direction for lateral surface
          // coupling
          parameter<int>("NUMPATCH_AXI",
              {.description = "Number of integration patches per 1D element in axial direction for "
                              "lateral surface coupling",
                  .default_value = 1}),

          // Number of integration patches per 1D element in radial direction for lateral surface
          // coupling
          parameter<int>("NUMPATCH_RAD",
              {.description =
                      "Number of integration patches per 1D element in radial direction for "
                      "lateral surface coupling",
                  .default_value = 1}),

          // Flag if blood vessel volume fraction should be output
          parameter<bool>("OUTPUT_BLOODVESSELVOLFRAC",
              {.description = "Flag if output of blood vessel volume fraction should be calculated",
                  .default_value = false}),

          // Flag if summary of coupling-pairs should be printed
          parameter<bool>("PRINT_OUT_SUMMARY_PAIRS",
              {.description = "Flag if summary of coupling-pairs should be printed",
                  .default_value = false}),

          // Flag if free-hanging elements (after blood vessel collapse) should be deleted
          parameter<bool>("DELETE_FREE_HANGING_ELES",
              {.description = "Flag if free-hanging elements (after blood vessel collapse) should "
                              "be deleted",
                  .default_value = false}),

          // components whose size is smaller than this fraction of the total network size are also
          // deleted
          parameter<double>("DELETE_SMALL_FREE_HANGING_COMPS",
              {.description =
                      "Small connected components whose size is smaller than this fraction of the "
                      "overall network size are additionally deleted (a valid choice of this "
                      "parameter should lie between 0 and 1)",
                  .default_value = -1.0})},
      {.defaultable = true});
}

FOUR_C_NAMESPACE_CLOSE