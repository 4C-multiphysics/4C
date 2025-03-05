// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_inpar_fsi.hpp"

#include "4C_fem_condition_definition.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*/
void Inpar::FSI::set_valid_parameters(std::map<std::string, Core::IO::InputSpec>& list)
{
  using Teuchos::tuple;
  using namespace Core::IO::InputSpecBuilders;

  Core::Utils::SectionSpecs fsidyn{"FSI DYNAMIC"};

  Teuchos::Tuple<std::string, 21> name;
  Teuchos::Tuple<FsiCoupling, 21> label;

  name[0] = "basic_sequ_stagg";
  label[0] = fsi_basic_sequ_stagg;
  name[1] = "iter_stagg_fixed_rel_param";
  label[1] = fsi_iter_stagg_fixed_rel_param;
  name[2] = "iter_stagg_AITKEN_rel_param";
  label[2] = fsi_iter_stagg_AITKEN_rel_param;
  name[3] = "iter_stagg_steep_desc";
  label[3] = fsi_iter_stagg_steep_desc;
  name[4] = "iter_stagg_NLCG";
  label[4] = fsi_iter_stagg_NLCG;
  name[5] = "iter_stagg_MFNK_FD";
  label[5] = fsi_iter_stagg_MFNK_FD;
  name[6] = "iter_stagg_MFNK_FSI";
  label[6] = fsi_iter_stagg_MFNK_FSI;
  name[7] = "iter_stagg_MPE";
  label[7] = fsi_iter_stagg_MPE;
  name[8] = "iter_stagg_RRE";
  label[8] = fsi_iter_stagg_RRE;
  name[9] = "iter_monolithicfluidsplit";
  label[9] = fsi_iter_monolithicfluidsplit;
  name[10] = "iter_monolithicstructuresplit";
  label[10] = fsi_iter_monolithicstructuresplit;
  name[11] = "iter_xfem_monolithic";
  label[11] = fsi_iter_xfem_monolithic;
  name[12] = "iter_mortar_monolithicstructuresplit";
  label[12] = fsi_iter_mortar_monolithicstructuresplit;
  name[13] = "iter_mortar_monolithicfluidsplit";
  label[13] = fsi_iter_mortar_monolithicfluidsplit;
  name[14] = "iter_fluidfluid_monolithicstructuresplit";
  label[14] = fsi_iter_fluidfluid_monolithicstructuresplit;
  name[15] = "iter_fluidfluid_monolithicfluidsplit";
  label[15] = fsi_iter_fluidfluid_monolithicfluidsplit;
  name[16] = "iter_fluidfluid_monolithicstructuresplit_nonox";
  label[16] = fsi_iter_fluidfluid_monolithicstructuresplit_nonox;
  name[17] = "iter_fluidfluid_monolithicfluidsplit_nonox";
  label[17] = fsi_iter_fluidfluid_monolithicfluidsplit_nonox;
  name[18] = "iter_sliding_monolithicfluidsplit";
  label[18] = fsi_iter_sliding_monolithicfluidsplit;
  name[19] = "iter_sliding_monolithicstructuresplit";
  label[19] = fsi_iter_sliding_monolithicstructuresplit;
  name[20] = "iter_mortar_monolithicfluidsplit_saddlepoint";
  label[20] = fsi_iter_mortar_monolithicfluidsplit_saddlepoint;


  Core::Utils::string_to_integral_parameter<FsiCoupling>("COUPALGO", "iter_stagg_AITKEN_rel_param",
      "Iteration Scheme over the fields", name, label, fsidyn);

  std::string debugoutput_doc =
      "Output of unconverged interface values during FSI iteration. There will be a new control "
      "file for each time step. This might be helpful to understand the coupling iteration.";
  fsidyn.specs.emplace_back(
      parameter<bool>("DEBUGOUTPUT", {.description = debugoutput_doc, .default_value = false}));
  fsidyn.specs.emplace_back(parameter<bool>("MATCHGRID_FLUIDALE",
      {.description = "is matching grid (fluid-ale)", .default_value = true}));

  fsidyn.specs.emplace_back(parameter<bool>("MATCHGRID_STRUCTALE",
      {.description = "is matching grid (structure-ale)", .default_value = true}));

  fsidyn.specs.emplace_back(parameter<bool>("MATCHALL",
      {.description = "is matching grid (fluid-ale) and is full fluid-ale (without euler part)",
          .default_value = true}));

  fsidyn.specs.emplace_back(parameter<double>(
      "MAXTIME", {.description = "Total simulation time", .default_value = 1000.0}));
  Core::Utils::int_parameter("NUMSTEP", 200, "Total number of Timesteps", fsidyn);

  Core::Utils::int_parameter("RESTARTEVERY", 1, "Increment for writing restart", fsidyn);

  fsidyn.specs.emplace_back(parameter<bool>(
      "RESTART_FROM_PART_FSI", {.description = "restart from partitioned fsi (e.g. from prestress "
                                               "calculations) instead of monolithic fsi",
                                   .default_value = false}));

  fsidyn.specs.emplace_back(parameter<bool>("SECONDORDER",
      {.description = "Second order displacement-velocity conversion at the interface.",
          .default_value = false}));

  Core::Utils::string_to_integral_parameter<Inpar::FSI::SlideALEProj>("SLIDEALEPROJ", "None",
      "Projection method to use for sliding FSI.",
      tuple<std::string>("None", "Curr", "Ref", "RotZ", "RotZSphere"),
      tuple<Inpar::FSI::SlideALEProj>(Inpar::FSI::ALEprojection_none,
          Inpar::FSI::ALEprojection_curr, Inpar::FSI::ALEprojection_ref,
          Inpar::FSI::ALEprojection_rot_z, Inpar::FSI::ALEprojection_rot_zsphere),
      fsidyn);

  fsidyn.specs.emplace_back(
      parameter<double>("TIMESTEP", {.description = "Time increment dt", .default_value = 0.1}));

  Core::Utils::int_parameter("RESULTSEVERY", 1, "Increment for writing solution", fsidyn);

  Core::Utils::string_to_integral_parameter<Inpar::FSI::Verbosity>("VERBOSITY", "full",
      "Verbosity of the FSI problem.", tuple<std::string>("full", "medium", "low", "subproblem"),
      tuple<Inpar::FSI::Verbosity>(Inpar::FSI::verbosity_full, Inpar::FSI::verbosity_medium,
          Inpar::FSI::verbosity_low, Inpar::FSI::verbosity_subproblem),
      fsidyn);

  fsidyn.move_into_collection(list);

  /*----------------------------------------------------------------------*/
  /* parameters for time step size adaptivity in fsi dynamics */
  Core::Utils::SectionSpecs fsiadapt{fsidyn, "TIMEADAPTIVITY"};

  Core::Utils::int_parameter("ADAPTSTEPMAX", 5,
      "Maximum number of repetitions of one time step for adapting/reducing the time step size "
      "(>0)",
      fsiadapt);

  Core::Utils::string_to_integral_parameter<Inpar::FSI::FluidMethod>("AUXINTEGRATORFLUID", "AB2",
      "Method for error estimation in the fluid field",
      tuple<std::string>("None", "ExplicitEuler", "AB2"),
      tuple<Inpar::FSI::FluidMethod>(Inpar::FSI::timada_fld_none, Inpar::FSI::timada_fld_expleuler,
          Inpar::FSI::timada_fld_adamsbashforth2),
      fsiadapt);

  fsiadapt.specs.emplace_back(parameter<std::string>("AVERAGINGDT",
      {.description = "Averaging of time step sizes in case of increasing time step "
                      "size.\nParameters are ordered from most recent weight to the most historic "
                      "one.\nNumber of parameters determines the number of previous time steps "
                      "that are involved\nin the averaging procedure.",
          .default_value = "0.3 0.7"}));


  Core::Utils::string_to_integral_parameter<Inpar::FSI::DivContAct>("DIVERCONT", "stop",
      "What to do if nonlinear solver does not converge?",
      tuple<std::string>("stop", "continue", "halve_step", "revert_dt"),
      tuple<Inpar::FSI::DivContAct>(Inpar::FSI::divcont_stop, Inpar::FSI::divcont_continue,
          Inpar::FSI::divcont_halve_step, Inpar::FSI::divcont_revert_dt),
      fsiadapt);

  fsiadapt.specs.emplace_back(parameter<double>("DTMAX",
      {.description = "Limit maximally permitted time step size (>0)", .default_value = 0.1}));
  fsiadapt.specs.emplace_back(parameter<double>("DTMIN",
      {.description = "Limit minimally allowed time step size (>0)", .default_value = 1.0e-4}));

  fsiadapt.specs.emplace_back(parameter<double>("LOCERRTOLFLUID",
      {.description = "Tolerance for the norm of local velocity error", .default_value = 1.0e-3}));


  Core::Utils::int_parameter("NUMINCREASESTEPS", 0,
      "Number of consecutive steps that want to increase time step size before\n"
      "actually increasing it. Set 0 to deactivate this feature.",
      fsiadapt);

  fsiadapt.specs.emplace_back(parameter<double>(
      "SAFETYFACTOR", {.description = "This is a safety factor to scale theoretical optimal step "
                                      "size, \nshould be lower than 1 and must be larger than 0",
                          .default_value = 0.9}));

  fsiadapt.specs.emplace_back(parameter<double>("SIZERATIOMAX",
      {.description =
              "Limit maximally permitted change of\ntime step size compared to previous size (>0).",
          .default_value = 2.0}));
  fsiadapt.specs.emplace_back(parameter<double>("SIZERATIOMIN",
      {.description =
              "Limit minimally permitted change of\ntime step size compared to previous size (>0).",
          .default_value = 0.5}));

  fsiadapt.specs.emplace_back(parameter<bool>("TIMEADAPTON",
      {.description = "Activate or deactivate time step size adaptivity", .default_value = false}));

  fsiadapt.move_into_collection(list);

  /*--------------------------------------------------------------------------*/

  /*--------------------------------------------------------------------------*/
  /* parameters for monolithic FSI solvers */
  Core::Utils::SectionSpecs fsimono{fsidyn, "MONOLITHIC SOLVER"};

  Core::Utils::double_parameter("ADAPTIVEDIST", 0.0,
      "Required distance for adaptive convergence check in Newton-type FSI.\n"
      "This is the improvement we want to achieve in the linear extrapolation of the\n"
      "adaptive convergence check. Set to zero to avoid the adaptive check altogether.",
      fsimono);

  Core::Utils::double_parameter("BASETOL", 1e-3,
      "Basic tolerance for adaptive convergence check in monolithic FSI.\n"
      "This tolerance will be used for the linear solve of the FSI block system.\n"
      "The linear convergence test will always use the relative residual norm (AZ_r0).\n"
      "Not to be confused with the Newton tolerance (CONVTOL) that applies\n"
      "to the nonlinear convergence test using a absolute residual norm.",
      fsimono);

  fsimono.specs.emplace_back(parameter<double>(
      "CONVTOL", {.description = "Nonlinear tolerance for lung/constraint/fluid-fluid FSI",
                     .default_value = 1e-6}));  // ToDo remove

  fsimono.specs.emplace_back(parameter<bool>("ENERGYFILE",
      {.description = "Write artificial interface energy due to temporal discretization to file",
          .default_value = false}));

  fsimono.specs.emplace_back(parameter<bool>("FSIAMGANALYZE",
      {.description = "run analysis on fsiamg multigrid scheme", .default_value = false}));

  fsimono.specs.emplace_back(parameter<bool>(
      "INFNORMSCALING", {.description = "Scale Blocks with row infnorm?", .default_value = true}));

  Core::Utils::int_parameter(
      "ITEMAX", 100, "Maximum allowed number of nonlinear iterations", fsimono);

  Core::Utils::int_parameter("KRYLOV_ITEMAX", 1000, "Max Iterations for linear solver.", fsimono);

  Core::Utils::int_parameter("KRYLOV_SIZE", 50, "Size of Krylov Subspace.", fsimono);

  Core::Utils::string_to_integral_parameter<Inpar::FSI::LinearBlockSolver>("LINEARBLOCKSOLVER",
      "PreconditionedKrylov", "Linear block preconditioner for block system in monolithic FSI.",
      tuple<std::string>("PreconditionedKrylov", "LinalgSolver"),
      tuple<Inpar::FSI::LinearBlockSolver>(
          Inpar::FSI::PreconditionedKrylov, Inpar::FSI::LinalgSolver),
      fsimono);

  Core::Utils::int_parameter("LINEAR_SOLVER", -1,
      "Number of SOLVER block describing the linear solver and preconditioner", fsimono);

  // Iteration parameters for convergence check of newton loop
  // for implementations without NOX
  Core::Utils::string_to_integral_parameter<Inpar::FSI::ConvNorm>("NORM_INC", "Rel",
      "type of norm for primary variables convergence check",
      tuple<std::string>("Abs", "Rel", "Mix"),
      tuple<Inpar::FSI::ConvNorm>(
          Inpar::FSI::convnorm_abs, Inpar::FSI::convnorm_rel, Inpar::FSI::convnorm_mix),
      fsimono);

  // for implementations without NOX
  Core::Utils::string_to_integral_parameter<Inpar::FSI::ConvNorm>("NORM_RESF", "Rel",
      "type of norm for residual convergence check", tuple<std::string>("Abs", "Rel", "Mix"),
      tuple<Inpar::FSI::ConvNorm>(
          Inpar::FSI::convnorm_abs, Inpar::FSI::convnorm_rel, Inpar::FSI::convnorm_mix),
      fsimono);

  // for implementations without NOX
  Core::Utils::string_to_integral_parameter<Inpar::FSI::BinaryOp>("NORMCOMBI_RESFINC", "And",
      "binary operator to combine primary variables and residual force values",
      tuple<std::string>("And"), tuple<Inpar::FSI::BinaryOp>(Inpar::FSI::bop_and), fsimono);

  Core::Utils::int_parameter("PRECONDREUSE", 0,
      "Number of iterations in one time step reusing the preconditioner before rebuilding it",
      fsimono);

  fsimono.specs.emplace_back(parameter<bool>("REBUILDPRECEVERYSTEP",
      {.description = "Enforce rebuilding the preconditioner at the beginning of every time step",
          .default_value = true}));

  fsimono.specs.emplace_back(parameter<bool>("SHAPEDERIVATIVES",
      {.description =
              "Include linearization with respect to mesh movement in Navier Stokes equation.",
          .default_value = false}));

  fsimono.specs.emplace_back(parameter<bool>("SYMMETRICPRECOND",
      {.description = "Symmetric block GS preconditioner or ordinary GS", .default_value = false}));

  // monolithic preconditioner parameter

  fsimono.specs.emplace_back(parameter<std::string>("ALEPCOMEGA",
      {.description = "Relaxation factor for Richardson iteration on ale block in MFSI block "
                      "preconditioner\nFSIAMG: each number belongs to a "
                      "level\nPreconditiondKrylov: only first number is used for finest level",
          .default_value = "1.0 1.0 1.0 1.0"}));
  fsimono.specs.emplace_back(parameter<std::string>("ALEPCITER",
      {.description = "Number of Richardson iterations on ale block in MFSI block "
                      "preconditioner\nFSIAMG: each number belongs to a "
                      "level\nPreconditiondKrylov: only first number is used for finest level",
          .default_value = "1 1 1 1"}));
  fsimono.specs.emplace_back(parameter<std::string>("FLUIDPCOMEGA",
      {.description = "Relaxation factor for Richardson iteration on fluid block in MFSI block "
                      "preconditioner\nFSIAMG: each number belongs to a "
                      "level\nPreconditiondKrylov: only first number is used for finest level",
          .default_value = "1.0 1.0 1.0 1.0"}));
  fsimono.specs.emplace_back(parameter<std::string>("FLUIDPCITER",
      {.description = "Number of Richardson iterations on fluid block in MFSI block "
                      "preconditioner\nFSIAMG: each number belongs to a "
                      "level\nPreconditiondKrylov: only first number is used for finest level",
          .default_value = "1 1 1 1"}));
  fsimono.specs.emplace_back(parameter<std::string>("STRUCTPCOMEGA",
      {.description = "Relaxation factor for Richardson iteration on structural block in MFSI "
                      "block \npreconditioner\nFSIAMG: each number belongs to a "
                      "level\nPreconditiondKrylov: only first number is used for finest level",
          .default_value = "1.0 1.0 1.0 1.0"}));
  fsimono.specs.emplace_back(parameter<std::string>("STRUCTPCITER",
      {.description = "Number of Richardson iterations on structural block in MFSI block "
                      "preconditioner\nFSIAMG: each number belongs to a "
                      "level\nPreconditiondKrylov: only first number is used for finest level",
          .default_value = "1 1 1 1"}));

  fsimono.specs.emplace_back(parameter<std::string>("PCOMEGA",
      {.description = "Relaxation factor for Richardson iteration on whole MFSI block "
                      "preconditioner\nFSIAMG: each number belongs to a "
                      "level\nPreconditiondKrylov: only first number is used for finest level",
          .default_value = "1.0 1.0 1.0"}));
  fsimono.specs.emplace_back(parameter<std::string>("PCITER",
      {.description = "Number of Richardson iterations on whole MFSI block preconditioner\nFSIAMG: "
                      "each number belongs to a level\nPreconditiondKrylov: only first number is "
                      "used for finest level",
          .default_value = "1 1 1"}));

  fsimono.specs.emplace_back(parameter<std::string>(
      "BLOCKSMOOTHER", {.description = "Type of block smoother, can be BGS or Schur",
                           .default_value = "BGS BGS BGS"}));

  fsimono.specs.emplace_back(parameter<std::string>(
      "SCHUROMEGA", {.description = "Damping factor for Schur complement construction",
                        .default_value = "0.001 0.01 0.1"}));

  // tolerances for convergence check of nonlinear solver in monolithic FSI
  // structure displacements
  fsimono.specs.emplace_back(parameter<double>("TOL_DIS_RES_L2",
      {.description = "Absolute tolerance for structure displacement residual in L2-norm",
          .default_value = 1e-6}));
  fsimono.specs.emplace_back(parameter<double>("TOL_DIS_RES_INF",
      {.description = "Absolute tolerance for structure displacement residual in Inf-norm",
          .default_value = 1e-6}));
  fsimono.specs.emplace_back(parameter<double>("TOL_DIS_INC_L2",
      {.description = "Absolute tolerance for structure displacement increment in L2-norm",
          .default_value = 1e-6}));
  fsimono.specs.emplace_back(parameter<double>("TOL_DIS_INC_INF",
      {.description = "Absolute tolerance for structure displacement increment in Inf-norm",
          .default_value = 1e-6}));
  // interface tolerances
  fsimono.specs.emplace_back(parameter<double>(
      "TOL_FSI_RES_L2", {.description = "Absolute tolerance for interface residual in L2-norm",
                            .default_value = 1e-6}));
  fsimono.specs.emplace_back(parameter<double>(
      "TOL_FSI_RES_INF", {.description = "Absolute tolerance for interface residual in Inf-norm",
                             .default_value = 1e-6}));
  fsimono.specs.emplace_back(parameter<double>(
      "TOL_FSI_INC_L2", {.description = "Absolute tolerance for interface increment in L2-norm",
                            .default_value = 1e-6}));
  fsimono.specs.emplace_back(parameter<double>(
      "TOL_FSI_INC_INF", {.description = "Absolute tolerance for interface increment in Inf-norm",
                             .default_value = 1e-6}));
  // fluid pressure
  fsimono.specs.emplace_back(parameter<double>(
      "TOL_PRE_RES_L2", {.description = "Absolute tolerance for fluid pressure residual in L2-norm",
                            .default_value = 1e-6}));
  fsimono.specs.emplace_back(parameter<double>("TOL_PRE_RES_INF",
      {.description = "Absolute tolerance for fluid pressure residual in Inf-norm",
          .default_value = 1e-6}));
  fsimono.specs.emplace_back(parameter<double>("TOL_PRE_INC_L2",
      {.description = "Absolute tolerance for fluid pressure increment in L2-norm",
          .default_value = 1e-6}));
  fsimono.specs.emplace_back(parameter<double>("TOL_PRE_INC_INF",
      {.description = "Absolute tolerance for fluid pressure increment in Inf-norm",
          .default_value = 1e-6}));
  // fluid velocities
  fsimono.specs.emplace_back(parameter<double>(
      "TOL_VEL_RES_L2", {.description = "Absolute tolerance for fluid velocity residual in L2-norm",
                            .default_value = 1e-6}));
  fsimono.specs.emplace_back(parameter<double>("TOL_VEL_RES_INF",
      {.description = "Absolute tolerance for fluid velocity residual in Inf-norm",
          .default_value = 1e-6}));
  fsimono.specs.emplace_back(parameter<double>("TOL_VEL_INC_L2",
      {.description = "Absolute tolerance for fluid velocity increment in L2-norm",
          .default_value = 1e-6}));
  fsimono.specs.emplace_back(parameter<double>("TOL_VEL_INC_INF",
      {.description = "Absolute tolerance for fluid velocity increment in Inf-norm",
          .default_value = 1e-6}));

  fsimono.move_into_collection(list);

  /*----------------------------------------------------------------------*/
  /* parameters for partitioned FSI solvers */
  Core::Utils::SectionSpecs fsipart{fsidyn, "PARTITIONED SOLVER"};

  Core::Utils::double_parameter("BASETOL", 1e-3,
      "Basic tolerance for adaptive convergence check in monolithic FSI.\n"
      "This tolerance will be used for the linear solve of the FSI block system.\n"
      "The linear convergence test will always use the relative residual norm (AZ_r0).\n"
      "Not to be confused with the Newton tolerance (CONVTOL) that applies\n"
      "to the nonlinear convergence test using a absolute residual norm.",
      fsipart);

  fsipart.specs.emplace_back(parameter<double>("CONVTOL",
      {.description = "Tolerance for iteration over fields in case of partitioned scheme",
          .default_value = 1e-6}));

  std::vector<std::string> coupmethod_valid_input = {"mortar", "conforming", "immersed"};
  fsipart.specs.emplace_back(selection<std::string>("COUPMETHOD", coupmethod_valid_input,
      {.description = "Coupling Method mortar or conforming nodes at interface",
          .default_value = "conforming"}));

  Core::Utils::string_to_integral_parameter<Inpar::FSI::CoupVarPart>("COUPVARIABLE", "Displacement",
      "Coupling variable at the interface", tuple<std::string>("Displacement", "Force", "Velocity"),
      tuple<Inpar::FSI::CoupVarPart>(Inpar::FSI::CoupVarPart::disp, Inpar::FSI::CoupVarPart::force,
          Inpar::FSI::CoupVarPart::vel),
      fsipart);

  fsipart.specs.emplace_back(parameter<bool>("DIVPROJECTION",
      {.description = "Project velocity into divergence-free subspace for partitioned fsi",
          .default_value = false}));

  Core::Utils::int_parameter("ITEMAX", 100, "Maximum number of iterations over fields", fsipart);

  fsipart.specs.emplace_back(parameter<double>("MAXOMEGA",
      {.description = "largest omega allowed for Aitken relaxation (0.0 means no constraint)",
          .default_value = 0.0}));

  fsipart.specs.emplace_back(parameter<double>(
      "MINOMEGA", {.description = "smallest omega allowed for Aitken relaxation (default is -1.0)",
                      .default_value = -1.0}));


  Core::Utils::string_to_integral_parameter<Inpar::FSI::PartitionedCouplingMethod>("PARTITIONED",
      "DirichletNeumann", "Coupling strategies for partitioned FSI solvers.",
      tuple<std::string>(
          "DirichletNeumann", "DirichletNeumannSlideALE", "DirichletNeumannVolCoupl"),
      tuple<Inpar::FSI::PartitionedCouplingMethod>(Inpar::FSI::DirichletNeumann,
          Inpar::FSI::DirichletNeumannSlideale, Inpar::FSI::DirichletNeumannVolCoupl),
      fsipart);

  std::vector<std::string> predictor_valid_input = {
      "d(n)", "d(n)+dt*(1.5*v(n)-0.5*v(n-1))", "d(n)+dt*v(n)", "d(n)+dt*v(n)+0.5*dt^2*a(n)"};
  fsipart.specs.emplace_back(selection<std::string>("PREDICTOR", predictor_valid_input,
      {.description = "Predictor for interface displacements", .default_value = "d(n)"}));


  fsipart.specs.emplace_back(parameter<double>(
      "RELAX", {.description = "fixed relaxation parameter for partitioned FSI solvers",
                   .default_value = 1.0}));

  fsipart.move_into_collection(list);

  /* ----------------------------------------------------------------------- */
  Core::Utils::SectionSpecs constrfsi{fsidyn, "CONSTRAINT"};

  Core::Utils::string_to_integral_parameter<Inpar::FSI::PrecConstr>("PRECONDITIONER", "Simple",
      "preconditioner to use", tuple<std::string>("Simple", "Simplec"),
      tuple<Inpar::FSI::PrecConstr>(Inpar::FSI::Simple, Inpar::FSI::Simplec), constrfsi);
  Core::Utils::int_parameter("SIMPLEITER", 2, "Number of iterations for simple pc", constrfsi);
  constrfsi.specs.emplace_back(parameter<double>(
      "ALPHA", {.description = "alpha parameter for simple pc", .default_value = 0.8}));

  constrfsi.move_into_collection(list);
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
  surfsac.add_component(selection<std::string>(
      "field", {"structure", "ale"}, {.description = "field", .default_value = "structure"}));

  condlist.push_back(surfsac);

  /*--------------------------------------------------------------------*/
  // Additional coupling of structure and fluid volumes (for lung fsi)

  Core::Conditions::ConditionDefinition volsfv(
      "DESIGN STRUCTURE FLUID VOLUME COUPLING VOL CONDITIONS", "StructFluidVolCoupling",
      "StructFluidVolCoupling", Core::Conditions::StructFluidVolCoupling, false,
      Core::Conditions::geometry_type_volume);

  volsfv.add_component(parameter<int>("coupling_id"));
  volsfv.add_component(selection<std::string>(
      "field", {"structure", "ale"}, {.description = "field", .default_value = "structure"}));

  condlist.push_back(volsfv);
}

FOUR_C_NAMESPACE_CLOSE
