// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_inpar_fsi.hpp"

#include "4C_fem_condition_definition.hpp"
#include "4C_io_input_spec_builders.hpp"
FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*/
void Inpar::FSI::set_valid_parameters(std::map<std::string, Core::IO::InputSpec>& list)
{
  using namespace Core::IO::InputSpecBuilders;

  list["FSI DYNAMIC"] = group("FSI DYNAMIC",
      {

          deprecated_selection<FsiCoupling>("COUPALGO",
              {
                  {"basic_sequ_stagg", fsi_basic_sequ_stagg},
                  {"iter_stagg_fixed_rel_param", fsi_iter_stagg_fixed_rel_param},
                  {"iter_stagg_AITKEN_rel_param", fsi_iter_stagg_AITKEN_rel_param},
                  {"iter_stagg_steep_desc", fsi_iter_stagg_steep_desc},
                  {"iter_stagg_NLCG", fsi_iter_stagg_NLCG},
                  {"iter_stagg_MFNK_FD", fsi_iter_stagg_MFNK_FD},
                  {"iter_stagg_MFNK_FSI", fsi_iter_stagg_MFNK_FSI},
                  {"iter_stagg_MPE", fsi_iter_stagg_MPE},
                  {"iter_stagg_RRE", fsi_iter_stagg_RRE},
                  {"iter_monolithicfluidsplit", fsi_iter_monolithicfluidsplit},
                  {"iter_monolithicstructuresplit", fsi_iter_monolithicstructuresplit},
                  {"iter_xfem_monolithic", fsi_iter_xfem_monolithic},
                  {"iter_mortar_monolithicstructuresplit",
                      fsi_iter_mortar_monolithicstructuresplit},
                  {"iter_mortar_monolithicfluidsplit", fsi_iter_mortar_monolithicfluidsplit},
                  {"iter_fluidfluid_monolithicstructuresplit",
                      fsi_iter_fluidfluid_monolithicstructuresplit},
                  {"iter_fluidfluid_monolithicfluidsplit",
                      fsi_iter_fluidfluid_monolithicfluidsplit},
                  {"iter_fluidfluid_monolithicstructuresplit_nonox",
                      fsi_iter_fluidfluid_monolithicstructuresplit_nonox},
                  {"iter_fluidfluid_monolithicfluidsplit_nonox",
                      fsi_iter_fluidfluid_monolithicfluidsplit_nonox},
                  {"iter_sliding_monolithicfluidsplit", fsi_iter_sliding_monolithicfluidsplit},
                  {"iter_sliding_monolithicstructuresplit",
                      fsi_iter_sliding_monolithicstructuresplit},
                  {"iter_mortar_monolithicfluidsplit_saddlepoint",
                      fsi_iter_mortar_monolithicfluidsplit_saddlepoint},
              },
              {.description = "Iteration Scheme over the fields",
                  .default_value = fsi_iter_stagg_AITKEN_rel_param}),

          parameter<bool>("MATCHGRID_FLUIDALE",
              {.description = "is matching grid (fluid-ale)", .default_value = true}),

          parameter<bool>("MATCHGRID_STRUCTALE",
              {.description = "is matching grid (structure-ale)", .default_value = true}),

          parameter<bool>("MATCHALL",
              {.description =
                      "is matching grid (fluid-ale) and is full fluid-ale (without euler part)",
                  .default_value = true}),

          parameter<double>(
              "MAXTIME", {.description = "Total simulation time", .default_value = 1000.0}),
          parameter<int>(
              "NUMSTEP", {.description = "Total number of Timesteps", .default_value = 200}),

          parameter<int>(
              "RESTARTEVERY", {.description = "Increment for writing restart", .default_value = 1}),

          parameter<bool>("RESTART_FROM_PART_FSI",
              {.description = "restart from partitioned fsi (e.g. from prestress "
                              "calculations) instead of monolithic fsi",
                  .default_value = false}),

          parameter<bool>("SECONDORDER",
              {.description = "Second order displacement-velocity conversion at the interface.",
                  .default_value = false}),

          deprecated_selection<Inpar::FSI::SlideALEProj>("SLIDEALEPROJ",
              {
                  {"None", Inpar::FSI::ALEprojection_none},
                  {"Curr", Inpar::FSI::ALEprojection_curr},
                  {"Ref", Inpar::FSI::ALEprojection_ref},
                  {"RotZ", Inpar::FSI::ALEprojection_rot_z},
                  {"RotZSphere", Inpar::FSI::ALEprojection_rot_zsphere},
              },
              {.description = "Projection method to use for sliding FSI.",
                  .default_value = Inpar::FSI::ALEprojection_none}),


          parameter<double>("TIMESTEP", {.description = "Time increment dt", .default_value = 0.1}),

          parameter<int>("RESULTSEVERY",
              {.description = "Increment for writing solution", .default_value = 1}),

          deprecated_selection<Inpar::FSI::Verbosity>("VERBOSITY",
              {
                  {"full", Inpar::FSI::verbosity_full},
                  {"medium", Inpar::FSI::verbosity_medium},
                  {"low", Inpar::FSI::verbosity_low},
                  {"subproblem", Inpar::FSI::verbosity_subproblem},
              },
              {.description = "Verbosity of the FSI problem.",
                  .default_value = Inpar::FSI::verbosity_full})},
      {.defaultable =
              true}); /*----------------------------------------------------------------------*/
  /* parameters for time step size adaptivity in fsi dynamics */
  list["FSI DYNAMIC/TIMEADAPTIVITY"] = group("FSI DYNAMIC/TIMEADAPTIVITY",
      {

          parameter<int>("ADAPTSTEPMAX",
              {.description =
                      "Maximum number of repetitions of one time step for adapting/reducing the "
                      "time step size (>0)",
                  .default_value = 5}),

          deprecated_selection<Inpar::FSI::FluidMethod>("AUXINTEGRATORFLUID",
              {
                  {"None", Inpar::FSI::timada_fld_none},
                  {"ExplicitEuler", Inpar::FSI::timada_fld_expleuler},
                  {"AB2", Inpar::FSI::timada_fld_adamsbashforth2},
              },
              {.description = "Method for error estimation in the fluid field",
                  .default_value = Inpar::FSI::timada_fld_adamsbashforth2}),

          parameter<std::string>("AVERAGINGDT",
              {.description =
                      "Averaging of time step sizes in case of increasing time step "
                      "size.\nParameters are ordered from most recent weight to the most historic "
                      "one.\nNumber of parameters determines the number of previous time steps "
                      "that are involved\nin the averaging procedure.",
                  .default_value = "0.3 0.7"}),


          deprecated_selection<Inpar::FSI::DivContAct>("DIVERCONT",
              {
                  {"stop", Inpar::FSI::divcont_stop},
                  {"continue", Inpar::FSI::divcont_continue},
                  {"halve_step", Inpar::FSI::divcont_halve_step},
                  {"revert_dt", Inpar::FSI::divcont_revert_dt},
              },
              {.description = "What to do if nonlinear solver does not converge?",
                  .default_value = Inpar::FSI::divcont_stop}),

          parameter<double>(
              "DTMAX", {.description = "Limit maximally permitted time step size (>0)",
                           .default_value = 0.1}),
          parameter<double>("DTMIN", {.description = "Limit minimally allowed time step size (>0)",
                                         .default_value = 1.0e-4}),

          parameter<double>(
              "LOCERRTOLFLUID", {.description = "Tolerance for the norm of local velocity error",
                                    .default_value = 1.0e-3}),


          parameter<int>("NUMINCREASESTEPS",
              {.description =
                      "Number of consecutive steps that want to increase time step size before\n"
                      "actually increasing it. Set 0 to deactivate this feature.",
                  .default_value = 0}),

          parameter<double>("SAFETYFACTOR",
              {.description = "This is a safety factor to scale theoretical optimal step "
                              "size, \nshould be lower than 1 and must be larger than 0",
                  .default_value = 0.9}),

          parameter<double>(
              "SIZERATIOMAX", {.description = "Limit maximally permitted change of\ntime "
                                              "step size compared to previous size (>0).",
                                  .default_value = 2.0}),
          parameter<double>(
              "SIZERATIOMIN", {.description = "Limit minimally permitted change of\ntime "
                                              "step size compared to previous size (>0).",
                                  .default_value = 0.5}),

          parameter<bool>(
              "TIMEADAPTON", {.description = "Activate or deactivate time step size adaptivity",
                                 .default_value = false})},
      {.defaultable = true});

  /*--------------------------------------------------------------------------*/

  /*--------------------------------------------------------------------------*/
  /* parameters for monolithic FSI solvers */
  list["FSI DYNAMIC/MONOLITHIC SOLVER"] = group("FSI DYNAMIC/MONOLITHIC SOLVER",
      {

          parameter<double>("ADAPTIVEDIST",
              {.description =
                      "Required distance for adaptive convergence check in Newton-type FSI.\n"
                      "This is the improvement we want to achieve in the linear extrapolation of "
                      "the\n"
                      "adaptive convergence check. Set to zero to avoid the adaptive check "
                      "altogether.",
                  .default_value = 0.0}),

          parameter<double>("BASETOL",
              {.description =
                      "Basic tolerance for adaptive convergence check in monolithic FSI.\n"
                      "This tolerance will be used for the linear solve of the FSI block system.\n"
                      "The linear convergence test will always use the relative residual norm "
                      "(AZ_r0).\n"
                      "Not to be confused with the Newton tolerance (CONVTOL) that applies\n"
                      "to the nonlinear convergence test using a absolute residual norm.",
                  .default_value = 1e-3}),
          parameter<double>(
              "CONVTOL", {.description = "Nonlinear tolerance for lung/constraint/fluid-fluid FSI",
                             .default_value = 1e-6}),  // ToDo remove

          parameter<bool>("ENERGYFILE",
              {.description =
                      "Write artificial interface energy due to temporal discretization to file",
                  .default_value = false}),

          parameter<bool>("FSIAMGANALYZE",
              {.description = "run analysis on fsiamg multigrid scheme", .default_value = false}),

          parameter<bool>("INFNORMSCALING",
              {.description = "Scale Blocks with row infnorm?", .default_value = true}),

          parameter<int>("ITEMAX", {.description = "Maximum allowed number of nonlinear iterations",
                                       .default_value = 100}),
          parameter<int>("KRYLOV_ITEMAX",
              {.description = "Max Iterations for linear solver.", .default_value = 1000}),
          parameter<int>(
              "KRYLOV_SIZE", {.description = "Size of Krylov Subspace.", .default_value = 50}),


          parameter<Inpar::FSI::LinearBlockSolver>("LINEARBLOCKSOLVER",
              {.description = "Linear block preconditioner for block system in monolithic FSI.",
                  .default_value = Inpar::FSI::PreconditionedKrylov}),

          parameter<int>("LINEAR_SOLVER",
              {.description =
                      "Number of SOLVER block describing the linear solver and preconditioner",
                  .default_value = -1}),

          // Iteration parameters for convergence check of newton loop
          // for implementations without NOX
          deprecated_selection<Inpar::FSI::ConvNorm>("NORM_INC",
              {
                  {"Abs", Inpar::FSI::convnorm_abs},
                  {"Rel", Inpar::FSI::convnorm_rel},
                  {"Mix", Inpar::FSI::convnorm_mix},
              },
              {.description = "type of norm for primary variables convergence check",
                  .default_value = Inpar::FSI::convnorm_rel}),

          // for implementations without NOX
          deprecated_selection<Inpar::FSI::ConvNorm>("NORM_RESF",
              {
                  {"Abs", Inpar::FSI::convnorm_abs},
                  {"Rel", Inpar::FSI::convnorm_rel},
                  {"Mix", Inpar::FSI::convnorm_mix},
              },
              {.description = "type of norm for residual convergence check",
                  .default_value = Inpar::FSI::convnorm_rel}),

          // for implementations without NOX
          deprecated_selection<Inpar::FSI::BinaryOp>("NORMCOMBI_RESFINC",
              {
                  {"And", Inpar::FSI::bop_and},
              },
              {.description =
                      "binary operator to combine primary variables and residual force values",
                  .default_value = Inpar::FSI::bop_and}),

          parameter<int>(
              "PRECONDREUSE", {.description = "Number of iterations in one time step reusing the "
                                              "preconditioner before rebuilding it",
                                  .default_value = 0}),

          parameter<bool>("REBUILDPRECEVERYSTEP",
              {.description =
                      "Enforce rebuilding the preconditioner at the beginning of every time step",
                  .default_value = true}),

          parameter<bool>(
              "SHAPEDERIVATIVES", {.description = "Include linearization with respect to mesh "
                                                  "movement in Navier Stokes equation.",
                                      .default_value = false}),

          parameter<bool>("SYMMETRICPRECOND",
              {.description = "Symmetric block GS preconditioner or ordinary GS",
                  .default_value = false}),

          // monolithic preconditioner parameter
          parameter<std::string>("ALEPCOMEGA",
              {.description =
                      "Relaxation factor for Richardson iteration on ale block in MFSI block "
                      "preconditioner\nFSIAMG: each number belongs to a "
                      "level\nPreconditiondKrylov: only first number is used for finest level",
                  .default_value = "1.0 1.0 1.0 1.0"}),
          parameter<std::string>("ALEPCITER",
              {.description =
                      "Number of Richardson iterations on ale block in MFSI block "
                      "preconditioner\nFSIAMG: each number belongs to a "
                      "level\nPreconditiondKrylov: only first number is used for finest level",
                  .default_value = "1 1 1 1"}),
          parameter<std::string>("FLUIDPCOMEGA",
              {.description =
                      "Relaxation factor for Richardson iteration on fluid block in MFSI block "
                      "preconditioner\nFSIAMG: each number belongs to a "
                      "level\nPreconditiondKrylov: only first number is used for finest level",
                  .default_value = "1.0 1.0 1.0 1.0"}),
          parameter<std::string>("FLUIDPCITER",
              {.description =
                      "Number of Richardson iterations on fluid block in MFSI block "
                      "preconditioner\nFSIAMG: each number belongs to a "
                      "level\nPreconditiondKrylov: only first number is used for finest level",
                  .default_value = "1 1 1 1"}),
          parameter<std::string>("STRUCTPCOMEGA",
              {.description =
                      "Relaxation factor for Richardson iteration on structural block in MFSI "
                      "block \npreconditioner\nFSIAMG: each number belongs to a "
                      "level\nPreconditiondKrylov: only first number is used for finest level",
                  .default_value = "1.0 1.0 1.0 1.0"}),
          parameter<std::string>("STRUCTPCITER",
              {.description =
                      "Number of Richardson iterations on structural block in MFSI block "
                      "preconditioner\nFSIAMG: each number belongs to a "
                      "level\nPreconditiondKrylov: only first number is used for finest level",
                  .default_value = "1 1 1 1"}),

          parameter<std::string>("PCOMEGA",
              {.description =
                      "Relaxation factor for Richardson iteration on whole MFSI block "
                      "preconditioner\nFSIAMG: each number belongs to a "
                      "level\nPreconditiondKrylov: only first number is used for finest level",
                  .default_value = "1.0 1.0 1.0"}),
          parameter<std::string>("PCITER",
              {.description =
                      "Number of Richardson iterations on whole MFSI block preconditioner\nFSIAMG: "
                      "each number belongs to a level\nPreconditiondKrylov: only first number is "
                      "used for finest level",
                  .default_value = "1 1 1"}),

          parameter<std::string>(
              "BLOCKSMOOTHER", {.description = "Type of block smoother, can be BGS or Schur",
                                   .default_value = "BGS BGS BGS"}),

          parameter<std::string>(
              "SCHUROMEGA", {.description = "Damping factor for Schur complement construction",
                                .default_value = "0.001 0.01 0.1"}),

          // tolerances for convergence check of nonlinear solver in monolithic FSI
          // structure displacements
          parameter<double>("TOL_DIS_RES_L2",
              {.description = "Absolute tolerance for structure displacement residual in L2-norm",
                  .default_value = 1e-6}),
          parameter<double>("TOL_DIS_RES_INF",
              {.description = "Absolute tolerance for structure displacement residual in Inf-norm",
                  .default_value = 1e-6}),
          parameter<double>("TOL_DIS_INC_L2",
              {.description = "Absolute tolerance for structure displacement increment in L2-norm",
                  .default_value = 1e-6}),
          parameter<double>("TOL_DIS_INC_INF",
              {.description = "Absolute tolerance for structure displacement increment in Inf-norm",
                  .default_value = 1e-6}),
          // interface tolerances
          parameter<double>("TOL_FSI_RES_L2",
              {.description = "Absolute tolerance for interface residual in L2-norm",
                  .default_value = 1e-6}),
          parameter<double>("TOL_FSI_RES_INF",
              {.description = "Absolute tolerance for interface residual in Inf-norm",
                  .default_value = 1e-6}),
          parameter<double>("TOL_FSI_INC_L2",
              {.description = "Absolute tolerance for interface increment in L2-norm",
                  .default_value = 1e-6}),
          parameter<double>("TOL_FSI_INC_INF",
              {.description = "Absolute tolerance for interface increment in Inf-norm",
                  .default_value = 1e-6}),
          // fluid pressure
          parameter<double>("TOL_PRE_RES_L2",
              {.description = "Absolute tolerance for fluid pressure residual in L2-norm",
                  .default_value = 1e-6}),
          parameter<double>("TOL_PRE_RES_INF",
              {.description = "Absolute tolerance for fluid pressure residual in Inf-norm",
                  .default_value = 1e-6}),
          parameter<double>("TOL_PRE_INC_L2",
              {.description = "Absolute tolerance for fluid pressure increment in L2-norm",
                  .default_value = 1e-6}),
          parameter<double>("TOL_PRE_INC_INF",
              {.description = "Absolute tolerance for fluid pressure increment in Inf-norm",
                  .default_value = 1e-6}),
          // fluid velocities
          parameter<double>("TOL_VEL_RES_L2",
              {.description = "Absolute tolerance for fluid velocity residual in L2-norm",
                  .default_value = 1e-6}),
          parameter<double>("TOL_VEL_RES_INF",
              {.description = "Absolute tolerance for fluid velocity residual in Inf-norm",
                  .default_value = 1e-6}),
          parameter<double>("TOL_VEL_INC_L2",
              {.description = "Absolute tolerance for fluid velocity increment in L2-norm",
                  .default_value = 1e-6}),
          parameter<double>("TOL_VEL_INC_INF",
              {.description = "Absolute tolerance for fluid velocity increment in Inf-norm",
                  .default_value = 1e-6})},
      {.defaultable = true});

  /*----------------------------------------------------------------------*/
  /* parameters for partitioned FSI solvers */
  list["FSI DYNAMIC/PARTITIONED SOLVER"] = group("FSI DYNAMIC/PARTITIONED SOLVER",
      {

          parameter<double>("BASETOL",
              {.description =
                      "Basic tolerance for adaptive convergence check in monolithic FSI.\n"
                      "This tolerance will be used for the linear solve of the FSI block system.\n"
                      "The linear convergence test will always use the relative residual norm "
                      "(AZ_r0).\n"
                      "Not to be confused with the Newton tolerance (CONVTOL) that applies\n"
                      "to the nonlinear convergence test using a absolute residual norm.",
                  .default_value = 1e-3}),

          parameter<double>("CONVTOL",
              {.description = "Tolerance for iteration over fields in case of partitioned scheme",
                  .default_value = 1e-6}),


          deprecated_selection<std::string>("COUPMETHOD", {"mortar", "conforming", "immersed"},
              {.description = "Coupling Method mortar or conforming nodes at interface",
                  .default_value = "conforming"}),

          deprecated_selection<Inpar::FSI::CoupVarPart>("COUPVARIABLE",
              {
                  {"Displacement", Inpar::FSI::CoupVarPart::disp},
                  {"Force", Inpar::FSI::CoupVarPart::force},
                  {"Velocity", Inpar::FSI::CoupVarPart::vel},
              },
              {.description = "Coupling variable at the interface",
                  .default_value = Inpar::FSI::CoupVarPart::disp}),

          parameter<bool>("DIVPROJECTION",
              {.description = "Project velocity into divergence-free subspace for partitioned fsi",
                  .default_value = false}),

          parameter<int>("ITEMAX",
              {.description = "Maximum number of iterations over fields", .default_value = 100}),

          parameter<double>("MAXOMEGA",
              {.description =
                      "largest omega allowed for Aitken relaxation (0.0 means no constraint)",
                  .default_value = 0.0}),

          parameter<double>("MINOMEGA",
              {.description = "smallest omega allowed for Aitken relaxation (default is -1.0)",
                  .default_value = -1.0}),



          deprecated_selection<Inpar::FSI::PartitionedCouplingMethod>("PARTITIONED",
              {
                  {"DirichletNeumann", Inpar::FSI::DirichletNeumann},
                  {"DirichletNeumannSlideALE", Inpar::FSI::DirichletNeumannSlideale},
                  {"DirichletNeumannVolCoupl", Inpar::FSI::DirichletNeumannVolCoupl},
              },
              {.description = "Coupling strategies for partitioned FSI solvers.",
                  .default_value = Inpar::FSI::DirichletNeumann}),

          deprecated_selection<std::string>("PREDICTOR",
              {"d(n)", "d(n)+dt*(1.5*v(n)-0.5*v(n-1))", "d(n)+dt*v(n)",
                  "d(n)+dt*v(n)+0.5*dt^2*a(n)"},
              {.description = "Predictor for interface displacements", .default_value = "d(n)"}),


          parameter<double>(
              "RELAX", {.description = "fixed relaxation parameter for partitioned FSI solvers",
                           .default_value = 1.0})},
      {.defaultable = true});

  /* ----------------------------------------------------------------------- */
  list["FSI DYNAMIC/CONSTRAINT"] = group("FSI DYNAMIC/CONSTRAINT",
      {parameter<Inpar::FSI::PrecConstr>("PRECONDITIONER",
           {.description = "preconditioner to use", .default_value = Inpar::FSI::Simple}),
          parameter<int>("SIMPLEITER",
              {.description = "Number of iterations for simple pc", .default_value = 2}),
          parameter<double>(
              "ALPHA", {.description = "alpha parameter for simple pc", .default_value = 0.8})},
      {.defaultable = true});
}

/*----------------------------------------------------------------------------*/
void Inpar::FSI::set_valid_conditions(std::vector<Core::Conditions::ConditionDefinition>& condlist)
{
  using namespace Core::IO::InputSpecBuilders;

  Core::Conditions::ConditionDefinition linefsi("DESIGN FSI COUPLING LINE CONDITIONS",
      "FSICoupling", "FSI Coupling", Core::Conditions::FSICoupling, true,
      Core::Conditions::geometry_type_line);
  Core::Conditions::ConditionDefinition surffsi("DESIGN FSI COUPLING SURF CONDITIONS",
      "FSICoupling", "FSI Coupling", Core::Conditions::FSICoupling, true,
      Core::Conditions::geometry_type_surface);

  linefsi.add_component(parameter<int>("coupling_id"));
  surffsi.add_component(parameter<int>("coupling_id"));

  condlist.push_back(linefsi);
  condlist.push_back(surffsi);

  /*--------------------------------------------------------------------*/
  // FSI define centerdisp for sliding interfaces

  Core::Conditions::ConditionDefinition linefsicd("DESIGN FSI COUPLING CENTER DISP LINE CONDITIONS",
      "FSICouplingCenterDisp", "FSI Coupling Center Disp", Core::Conditions::FSICouplingCenterDisp,
      true, Core::Conditions::geometry_type_line);
  Core::Conditions::ConditionDefinition surffsicd("DESIGN FSI COUPLING CENTER DISP SURF CONDITIONS",
      "FSICouplingCenterDisp", "FSI Coupling Center Disp", Core::Conditions::FSICouplingCenterDisp,
      true, Core::Conditions::geometry_type_surface);

  condlist.push_back(linefsicd);
  condlist.push_back(surffsicd);

  /*--------------------------------------------------------------------*/
  // Additional coupling of structure and ale fields (for lung fsi)

  Core::Conditions::ConditionDefinition surfsac("DESIGN STRUCTURE ALE COUPLING SURF CONDITIONS",
      "StructAleCoupling", "StructAleCoupling", Core::Conditions::StructAleCoupling, true,
      Core::Conditions::geometry_type_surface);

  surfsac.add_component(parameter<int>("coupling_id"));
  surfsac.add_component(deprecated_selection<std::string>(
      "field", {"structure", "ale"}, {.description = "field", .default_value = "structure"}));

  condlist.push_back(surfsac);

  /*--------------------------------------------------------------------*/
  // Additional coupling of structure and fluid volumes (for lung fsi)

  Core::Conditions::ConditionDefinition volsfv(
      "DESIGN STRUCTURE FLUID VOLUME COUPLING VOL CONDITIONS", "StructFluidVolCoupling",
      "StructFluidVolCoupling", Core::Conditions::StructFluidVolCoupling, false,
      Core::Conditions::geometry_type_volume);

  volsfv.add_component(parameter<int>("coupling_id"));
  volsfv.add_component(deprecated_selection<std::string>(
      "field", {"structure", "ale"}, {.description = "field", .default_value = "structure"}));

  condlist.push_back(volsfv);
}

FOUR_C_NAMESPACE_CLOSE