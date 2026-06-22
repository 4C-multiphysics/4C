// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_structure_new_timint_basedataio.hpp"

#include "4C_global_data.hpp"
#include "4C_io_control.hpp"
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
      params_runtime_vtk_output_(nullptr),
      params_runtime_vtp_output_(nullptr),
      params_monitor_dbc_(nullptr),
      energyfile_(nullptr),
      gmsh_out_(false),
      printlogo_(false),
      printiter_(false),
      writesurfactant_(false),
      writestate_(false),
      writejac2matlab_(false),
      firstoutputofrun_(false),
      printscreen_(-1),
      outputcounter_(-1),
      writerestartevery_(-1),
      writeresultsevery_(-1),
      writeenergyevery_(-1),
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
    output_per_rank_eval_time_ = ioparams.get<bool>("PER_RANK_EVAL_TIME");

    // build params container for monitoring reaction forces
    params_monitor_dbc_ =
        std::make_shared<ParamsMonitorDBC>(ioparams.sublist("MONITOR STRUCTURE DBC"));

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



FOUR_C_NAMESPACE_CLOSE
