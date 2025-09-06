// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_structure_new_timint_basedataio.hpp"

#include "4C_global_data.hpp"
#include "4C_io_control.hpp"
#include "4C_io_every_iteration_writer.hpp"
#include "4C_solver_nonlin_nox_aux.hpp"
#include "4C_solver_nonlin_nox_linesearch_generic.hpp"
#include "4C_solver_nonlin_nox_linesearch_prepostoperator.hpp"
#include "4C_structure_new_timint_basedataio_monitor_dbc.hpp"
#include "4C_structure_new_timint_basedataio_runtime_vtk_output.hpp"
#include "4C_structure_new_timint_basedataio_runtime_vtp_output.hpp"

#include <NOX_Solver_Generic.H>
#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN

namespace
{
  inline bool determine_write_output(int step, int offset, int write_every)
  {
    return (step + offset) % write_every == 0;
  }
}  // namespace

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Solid::TimeInt::BaseDataIO::BaseDataIO()
    : isinit_(false),
      issetup_(false),
      output_(nullptr),
      writer_every_iter_(nullptr),
      params_runtime_vtk_output_(nullptr),
      params_runtime_vtp_output_(nullptr),
      params_monitor_dbc_(nullptr),
      energyfile_(nullptr),
      gmsh_out_(false),
      printlogo_(false),
      printiter_(false),
      outputeveryiter_(false),
      writesurfactant_(false),
      writestate_(false),
      writejac2matlab_(false),
      firstoutputofrun_(false),
      printscreen_(-1),
      outputcounter_(-1),
      writerestartevery_(-1),
      writeresultsevery_(-1),
      writeenergyevery_(-1),
      lastwrittenresultsstep_(-1),
      writestress_(Inpar::Solid::stress_none),
      writestrain_(Inpar::Solid::strain_none),
      writeplstrain_(Inpar::Solid::strain_none),
      conditionnumbertype_(Inpar::Solid::ConditionNumber::none)
{
  // empty constructor
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::BaseDataIO::init(const Teuchos::ParameterList& ioparams,
    const Teuchos::ParameterList& sdynparams, const Teuchos::ParameterList& xparams,
    std::shared_ptr<Core::IO::DiscretizationWriter> output)
{
  // We have to call setup() after init()
  issetup_ = false;

  // ---------------------------------------------------------------------------
  // initialize the printing and output parameters
  // ---------------------------------------------------------------------------
  {
    output_ = output;
    printscreen_ = ioparams.get<int>("STDOUTEVERY");
    printlogo_ = printscreen_ > 0;
    gmsh_out_ = ioparams.get<bool>("OUTPUT_GMSH");
    printiter_ = true;
    p_io_every_iteration_ =
        std::make_shared<Teuchos::ParameterList>(ioparams.sublist("EVERY ITERATION"));
    outputeveryiter_ = p_io_every_iteration_->get<bool>("OUTPUT_EVERY_ITER");
    writerestartevery_ = sdynparams.get<int>("RESTARTEVERY");
    writetimestepoffset_ = sdynparams.get<int>("OUTPUT_STEP_OFFSET");
    writestate_ = ioparams.get<bool>("STRUCT_DISP");
    writejac2matlab_ = ioparams.get<bool>("STRUCT_JACOBIAN_MATLAB");
    conditionnumbertype_ = ioparams.get<Inpar::Solid::ConditionNumber>("STRUCT_CONDITION_NUMBER");
    firstoutputofrun_ = true;
    writeresultsevery_ = sdynparams.get<int>("RESULTSEVERY");
    writestress_ = Teuchos::getIntegralValue<Inpar::Solid::StressType>(ioparams, "STRUCT_STRESS");
    writestrain_ = Teuchos::getIntegralValue<Inpar::Solid::StrainType>(ioparams, "STRUCT_STRAIN");
    writeplstrain_ =
        Teuchos::getIntegralValue<Inpar::Solid::StrainType>(ioparams, "STRUCT_PLASTIC_STRAIN");
    writeenergyevery_ = sdynparams.get<int>("RESEVERYERGY");
    writesurfactant_ = ioparams.get<bool>("STRUCT_SURFACTANT");

    // build params container for monitoring reaction forces
    params_monitor_dbc_ = std::make_shared<ParamsMonitorDBC>();
    params_monitor_dbc_->init(ioparams.sublist("MONITOR STRUCTURE DBC"));
    params_monitor_dbc_->setup();

    // check whether VTK output at runtime is desired
    if (ioparams.sublist("RUNTIME VTK OUTPUT").get<int>("INTERVAL_STEPS") != -1)
    {
      params_runtime_vtk_output_ = std::make_shared<ParamsRuntimeOutput>();

      params_runtime_vtk_output_->init(ioparams.sublist("RUNTIME VTK OUTPUT"));
      params_runtime_vtk_output_->setup();
    }

    // check whether VTP output at runtime is desired
    if (ioparams.sublist("RUNTIME VTP OUTPUT STRUCTURE").get<int>("INTERVAL_STEPS") != -1)
    {
      params_runtime_vtp_output_ = std::make_shared<ParamsRuntimeVtpOutput>();

      params_runtime_vtp_output_->init(ioparams.sublist("RUNTIME VTP OUTPUT STRUCTURE"));
      params_runtime_vtp_output_->setup();
    }
  }

  isinit_ = true;
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::BaseDataIO::setup()
{
  // safety check
  FOUR_C_ASSERT(is_init(), "init() has not been called, yet!");

  if (outputeveryiter_) writer_every_iter_ = std::make_shared<Core::IO::EveryIterationWriter>();

  issetup_ = true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::BaseDataIO::check_init_setup() const
{
  FOUR_C_ASSERT(is_init() and is_setup(), "Call init() and setup() first!");
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::BaseDataIO::init_setup_every_iteration_writer(
    Core::IO::EveryIterationWriterInterface* interface, Teuchos::ParameterList& p_nox)
{
  if (not outputeveryiter_) return;

  writer_every_iter_->init(output_.get(), interface, *p_io_every_iteration_);
  writer_every_iter_->setup();

  // insert the every_iter output writer as ppo for the solver object
  Teuchos::ParameterList& p_sol_opt = p_nox.sublist("Solver Options");

  Teuchos::RCP<::NOX::Observer> prepost_solver_ptr =
      Teuchos::make_rcp<NOX::Nln::Solver::PrePostOp::TimeInt::WriteOutputEveryIteration>(
          *writer_every_iter_);

  NOX::Nln::Aux::add_to_pre_post_op_vector(p_sol_opt, prepost_solver_ptr);

  // insert the every_iter output writer as ppo for the linesearch object
  Teuchos::ParameterList& p_linesearch = p_nox.sublist("Line Search");

  // Get the current map. If there is no map, return a new empty one. (reference)
  NOX::Nln::LineSearch::PrePostOperator::map& prepostls_map =
      NOX::Nln::LineSearch::PrePostOperator::get_map(p_linesearch);

  // insert/replace the old pointer in the map
  prepostls_map[NOX::Nln::LineSearch::prepost_output_every_iter] =
      Teuchos::rcp_dynamic_cast<NOX::Nln::Abstract::PrePostOperator>(prepost_solver_ptr);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::BaseDataIO::setup_energy_output_file()
{
  if (!energyfile_)
  {
    std::string energy_file_name =
        Global::Problem::instance()->output_control_file()->file_name() + "_energy.csv";

    energyfile_ = std::make_shared<std::ofstream>(energy_file_name.c_str());
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Solid::TimeInt::BaseDataIO::write_results_for_this_step(const int step) const
{
  if (step < 0) FOUR_C_THROW("The variable step is not allowed to be negative.");
  return is_write_results_enabled() and determine_write_output(step, get_write_timestep_offset(),
                                            get_write_results_every_n_step());
}

bool Solid::TimeInt::BaseDataIO::is_write_results_enabled() const
{
  return get_write_results_every_n_step() > 0;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Solid::TimeInt::BaseDataIO::write_runtime_vtk_results_for_this_step(const int step) const
{
  if (step < 0) FOUR_C_THROW("The variable step is not allowed to be negative.");
  return (is_runtime_output_enabled() &&
          determine_write_output(step, get_runtime_output_params()->output_step_offset(),
              get_runtime_output_params()->output_interval_in_steps()));
}

bool Solid::TimeInt::BaseDataIO::is_runtime_output_enabled() const
{
  return get_runtime_output_params() != nullptr;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Solid::TimeInt::BaseDataIO::write_runtime_vtp_results_for_this_step(const int step) const
{
  if (step < 0) FOUR_C_THROW("The variable step is not allowed to be negative.");
  return (get_runtime_vtp_output_params() != nullptr &&
          determine_write_output(step, get_runtime_output_params()->output_step_offset(),
              get_runtime_output_params()->output_interval_in_steps()));
}


bool Solid::TimeInt::BaseDataIO::should_write_restart_for_step(const int step) const
{
  return get_write_restart_every_n_step() &&
         determine_write_output(
             step, get_write_timestep_offset(), get_write_restart_every_n_step()) &&
         step != 0;
}


bool Solid::TimeInt::BaseDataIO::should_write_reaction_forces_for_this_step(const int step) const
{
  return get_monitor_dbc_params()->output_interval_in_steps() > 0 &&
         determine_write_output(step, get_write_timestep_offset(),
             get_monitor_dbc_params()->output_interval_in_steps());
}


bool Solid::TimeInt::BaseDataIO::should_write_energy_for_this_step(const int step) const
{
  return get_write_energy_every_n_step() > 0 &&
         determine_write_output(step, get_write_timestep_offset(), get_write_energy_every_n_step());
}

int Solid::TimeInt::BaseDataIO::get_last_written_results() const { return lastwrittenresultsstep_; }

void Solid::TimeInt::BaseDataIO::set_last_written_results(const int step)
{
  lastwrittenresultsstep_ = step;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
NOX::Nln::Solver::PrePostOp::TimeInt::WriteOutputEveryIteration::WriteOutputEveryIteration(
    Core::IO::EveryIterationWriter& every_iter_writer)
    : every_iter_writer_(every_iter_writer)
{ /* empty */
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void NOX::Nln::Solver::PrePostOp::TimeInt::WriteOutputEveryIteration::runPreSolve(
    const ::NOX::Solver::Generic& solver)
{
  every_iter_writer_.init_newton_iteration();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void NOX::Nln::Solver::PrePostOp::TimeInt::WriteOutputEveryIteration::runPostIterate(
    const ::NOX::Solver::Generic& solver)
{
  const int newton_iteration = solver.getNumIterations();
  every_iter_writer_.add_newton_iteration(newton_iteration);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void NOX::Nln::Solver::PrePostOp::TimeInt::WriteOutputEveryIteration::run_pre_modify_step_length(
    const ::NOX::Solver::Generic& solver, const ::NOX::LineSearch::Generic& linesearch)
{
  const int newton_iteration = solver.getNumIterations();
  const int ls_iteration =
      dynamic_cast<const NOX::Nln::LineSearch::Generic&>(linesearch).get_num_iterations();
  every_iter_writer_.add_line_search_iteration(newton_iteration, ls_iteration);
}

FOUR_C_NAMESPACE_CLOSE
