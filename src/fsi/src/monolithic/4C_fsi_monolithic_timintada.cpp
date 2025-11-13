// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_adapter_ale_fsi.hpp"
#include "4C_adapter_fld_fluid_fsi.hpp"
#include "4C_adapter_str_fsi_timint_adaptive.hpp"
#include "4C_fluid_utils_mapextractor.hpp"
#include "4C_fsi_monolithic.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_fsi.hpp"
#include "4C_io_control.hpp"
#include "4C_io_pstream.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_structure_aux.hpp"
#include "4C_timestepping_mstep.hpp"

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_StandardParameterEntryValidators.hpp>
#include <Teuchos_TimeMonitor.hpp>

#include <memory>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void FSI::Monolithic::init_tim_int_ada(const Teuchos::ParameterList& fsidyn)
{
  // access to structural time adaptivity parameters
  const Teuchos::ParameterList& sdyn = Global::Problem::instance()->structural_dynamic_params();
  const Teuchos::ParameterList& sada = sdyn.sublist("TIMEADAPTIVITY");

  // access to FSI time adaptivity parameters
  const Teuchos::ParameterList& fsiada = fsidyn.sublist("TIMEADAPTIVITY");

  // initialize member variables
  adaptstep_ = 0;
  accepted_ = false;

  adareason_ = "none";

  // L2-norms of estimation of temporal discretization errors
  strnorm_ = 0.0;
  flnorm_ = 0.0;
  strfsinorm_ = 0.0;
  flfsinorm_ = 0.0;
  strinnernorm_ = 0.0;
  flinnernorm_ = 0.0;

  // L-inf-norms of estimation of temporal discretization errors
  strinfnorm_ = 0.0;
  flinfnorm_ = 0.0;
  strinffsinorm_ = 0.0;
  flinffsinorm_ = 0.0;
  strinfinnernorm_ = 0.0;
  flinfinnernorm_ = 0.0;

  // time step sizes calculated according to the 6 available L2-norms
  dtstr_ = 0.0;
  dtfl_ = 0.0;
  dtstrfsi_ = 0.0;
  dtflfsi_ = 0.0;
  dtstrinner_ = 0.0;
  dtflinner_ = 0.0;
  dtnonlinsolver_ = 0.0;

  dtminused_ = false;

  numincreasesteps_ = fsiada.get<int>("NUMINCREASESTEPS");

  // read weights for averaging strategy in case of increasing time step size
  {
    double word;
    std::istringstream avgweightstream(Teuchos::getNumericStringParameter(fsiada, "AVERAGINGDT"));
    while (avgweightstream >> word) avgweights_.push_back(word);

    // check if weights for averaging make sense
    double sum = 0.0;
    for (std::vector<double>::iterator it = avgweights_.begin(); it < avgweights_.end(); ++it)
      sum += (*it);
    if (sum < 0.99)
    {
      FOUR_C_THROW("Sum of weights for dt: {} < 1.0", sum);
    }
    if (sum > 1.01)
    {
      FOUR_C_THROW("Sum of weights for dt: {} > 1.0", sum);
    }
  }

  dt_ = TimeStepping::TimIntMStep(-avgweights_.size(), 1, 0.0);
  dt_.set_step(1, dt());

  //----------------------------------------------------------------------------
  // write adaptivity file
  //----------------------------------------------------------------------------
  std::string fileada = Global::Problem::instance()->output_control_file()->file_name();
  fileada.append(".adaptivity");
  logada_ = std::make_shared<std::ofstream>(fileada.c_str());

  //----------------------------------------------------------------------------
  // set algorithmic parameters
  //----------------------------------------------------------------------------
  dtmax_ = fsiada.get<double>("DTMAX");
  dtmin_ = fsiada.get<double>("DTMIN");

  errtolstr_ = sada.get<double>("LOCERRTOL");
  errtolfl_ = fsiada.get<double>("LOCERRTOLFLUID");

  //----------------------------------------------------------------------------
  // check on which fields time adaptivity should be based on
  //----------------------------------------------------------------------------
  if (not(Teuchos::getIntegralValue<Inpar::Solid::TimAdaKind>(sada, "KIND") ==
          Inpar::Solid::timada_kind_none))
    isadastructure_ = true;

  if (not(Teuchos::getIntegralValue<Inpar::FSI::FluidMethod>(fsiada, "AUXINTEGRATORFLUID") ==
          Inpar::FSI::timada_fld_none))
    isadafluid_ = true;

  // get error action strategy from input file
  const auto erroractionstrategy =
      Teuchos::getIntegralValue<Inpar::FSI::DivContAct>(fsiada, "DIVERCONT");
  if (erroractionstrategy != Inpar::FSI::divcont_stop and
      erroractionstrategy != Inpar::FSI::divcont_continue)
    isadasolver_ = true;

  flmethod_ = fluid_field()->get_tim_ada_method_name();

  //----------------------------------------------------------------------------
  // Handling of Dirichlet BCs in error estimation
  //----------------------------------------------------------------------------
  // Create intersection of fluid DOFs that hold a Dirichlet boundary condition
  // and are located at the FSI interface.
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> intersectionmapsfluid;
  intersectionmapsfluid.push_back(fluid_field()->get_dbc_map_extractor()->cond_map());
  intersectionmapsfluid.push_back(fluid_field()->interface()->fsi_cond_map());
  std::shared_ptr<Core::LinAlg::Map> intersectionmapfluid =
      Core::LinAlg::MultiMapExtractor::intersect_maps(intersectionmapsfluid);

  // store number of interface DOFs subject to Dirichlet BCs on structure and fluid side of the
  // interface
  numflfsidbcdofs_ = intersectionmapfluid->num_global_elements();

  //----------------------------------------------------------------------------
  // Check whether input parameters make sense
  //----------------------------------------------------------------------------
  const double safetyfac = fsiada.get<double>("SAFETYFACTOR");
  if (safetyfac > 1.0)
    FOUR_C_THROW(
        "SAFETYFACTOR in FSI DYNAMIC/TIMEADAPTIVITY is {} > 1.0 and, "
        "thus, too large.",
        safetyfac);

  if (dtmax_ < dtmin_)
    FOUR_C_THROW("DTMAX = {} is not allowed to be smaller than DTMIN = {}.", dtmax_, dtmin_);

  /* safety check for BDF2 time integration in fluid
   * (cf. PhD Thesis [Bornemann, 2003, p. 61, eq. (3.40)]) */
  if (fluid_field()->tim_int_scheme() == Inpar::FLUID::timeint_bdf2 and
      fsiada.get<double>("SIZERATIOMAX") >= 1.0 + sqrt(2))
  {
    FOUR_C_THROW(
        "In case of BDF2 time integration, the maximum increase of "
        "two subsequent time step sizes is limited to be less than '1+sqrt(2)'"
        "due to stability reasons "
        "(cf. PhD Thesis [Bornemann, 2003, p. 61, eq. (3.40)]). "
        "Choose an appropriate value!");
  }

  return;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void FSI::Monolithic::timeloop_ada_dt(
    const std::shared_ptr<NOX::Nln::Interface::RequiredBase> interface)
{
  /*--------------------------------------------------------------------------*/
  /* Initialize some parameters                                               */
  /*--------------------------------------------------------------------------*/
  // get the FSI parameter list
  const Teuchos::ParameterList& fsidyn = Global::Problem::instance()->fsi_dynamic_params();

  // number of maximum allowed time step repetitions
  const int adaptstepmax = fsidyn.sublist("TIMEADAPTIVITY").get<int>("ADAPTSTEPMAX");
  /*--------------------------------------------------------------------------*/

  // resize MStep objects of structure needed for AB2 structural extrapolation of displacements
  structure_field()->resize_m_step_tim_ada();

#ifdef FOUR_C_ENABLE_ASSERTIONS
  // check whether all fields have the same time step size
  check_if_dts_same();
#endif

  prepare_timeloop();

  // the time loop
  while (not_finished())
  {
    prepare_adaptive_time_step();

    // time step adaptivity loop
    while (step_not_accepted() and (adaptstep_ < adaptstepmax) and (not dtminused_))
    {
      print_header_repeated_step();
      prepare_time_step();

      // Do the auxiliary step first
      time_step_auxiliary();

      // Do the time step with the marching time integrator
      time_step(interface);

      // adjust the time step size
      adapt_time_step_size();

      /* continue with the next time step if the step has been repeated with the minimum
       * time step size --> no more refinement is possible anyway!*/
      if (dtminused_)
      {
        if (Core::Communication::my_mpi_rank(get_comm()) == 0)
        {
          Core::IO::cout << "Time Step has already been calculated with minimum step size"
                         << " --> continue with next time step!"
                         << "\n";
        }

        accepted_ = true;
      }

      if (step_not_accepted())
      {
        if (adaptstep_ >= adaptstepmax)
        {
          if (not is_ada_solver())
          {
            if (Core::Communication::my_mpi_rank(get_comm()) == 0)
            {
              Core::IO::cout << "adaptstep_ = " << adaptstepmax
                             << " --> Max. number of adaption iterations is reached:"
                             << " continuing with next time step! "
                             << "\n";
            }
          }
          else if (is_ada_solver() and not(erroraction_ == FSI::Monolithic::erroraction_none or
                                           erroraction_ == FSI::Monolithic::erroraction_continue))
          {
            FOUR_C_THROW(
                "Nonlinear solver did not converge after {} repetitions of "
                "time step {}.",
                adaptstepmax, step());
          }
        }
        else
        {
          reset_step();
          reset_time();

          if (Core::Communication::my_mpi_rank(get_comm()) == 0)
          {
            Core::IO::cout << "Repeat current time step with dt = " << dt() << " based on "
                           << adareason_ << ".\n";
          }
        }
      }

#ifdef FOUR_C_ENABLE_ASSERTIONS
      // check whether all fields have the same time step size
      check_if_dts_same();
#endif

      print_adaptivity_summary();
    }
    constexpr bool force_prepare = false;
    prepare_output(force_prepare);
    update();              // Update in single fields
    update_dt_past(dt());  // Update of adaptive time step sizes
    output();

    write_ada_file();
  }

  return;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void FSI::Monolithic::prepare_adaptive_time_step()
{
  // reset from last time step
  dtminused_ = false;
  adaptstep_ = 0;
  accepted_ = false;

  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
  {
    Core::IO::cout << "\n"
                   << "+++++++++++++++++++++++++++++NEW TIME STEP+++++++++++++++++++++++++++++";
  }

  return;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void FSI::Monolithic::print_header_repeated_step() const
{
  if (adaptstep_ != 0 and Core::Communication::my_mpi_rank(get_comm()) == 0)
  {
    Core::IO::cout << Core::IO::endl
                   << "__________REPEATING TIME STEP " << step() << " WITH DT = " << dt()
                   << " FOR THE " << adaptstep_ << ". TIME__________" << Core::IO::endl;
  }
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void FSI::Monolithic::write_ada_file_header() const
{
  // write to adaptivity file
  if (Core::Communication::my_mpi_rank(get_comm()) == 0 and (logada_))
  {
    // get string of type of auxiliary time integration scheme in structure field
    const Teuchos::ParameterList& sdyn = Global::Problem::instance()->structural_dynamic_params();
    const Teuchos::ParameterList& sada = sdyn.sublist("TIMEADAPTIVITY");
    const auto strmethod = Teuchos::getIntegralValue<Inpar::Solid::TimAdaKind>(sada, "KIND");

    // print the actual header
    (*logada_) << "Time Adaptivity in monolithic Fluid-Structure-Interaction:"
               << "\n"
               << " - Error estimation method in fluid field: " << flmethod_ << "\n"
               << " - Error estimation method in structure field: " << strmethod << "\n"
               << " - Error tolerance in fluid field: " << errtolfl_ << "\n"
               << " - Error tolerance in structure field: " << errtolstr_ << "\n"
               << " - Minimum allowed time step size DTMIN = " << dtmin_ << "\n"
               << " - Maximum allowed time step size DTMAX = " << dtmax_ << "\n \n"
               << std::right << std::setw(9) << "step" << std::right << std::setw(16) << "time"
               << std::right << std::setw(16) << "dt" << std::right << std::setw(16) << "#adaptiter"
               << std::right << std::setw(16) << "dt of field" << std::right << std::setw(16)
               << "L2-err str field" << std::right << std::setw(16) << "L2-err str inner"
               << std::right << std::setw(16) << "L2-err str fsi" << std::right << std::setw(16)
               << "L2-err fl field" << std::right << std::setw(16) << "L2-err fl inner"
               << std::right << std::setw(16) << "L2-err fl fsi" << std::right << std::setw(16)
               << "Linf str field" << std::right << std::setw(16) << "Linf str inner" << std::right
               << std::setw(16) << "Linf str fsi" << std::right << std::setw(16) << "Linf fl field"
               << std::right << std::setw(16) << "Linf fl inner" << std::right << std::setw(16)
               << "Linf fl fsi"
               << "\n\n";
  }
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void FSI::Monolithic::write_ada_file() const
{
  if (!logada_) FOUR_C_THROW("No access to adaptivity file!");

  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
  {
    (*logada_) << std::right << std::setw(9) << step() << std::right << std::setw(16) << time()
               << std::right << std::setw(16) << dt_past(1) << std::right << std::setw(16)
               << adaptstep_;

    // Who is responsible for the new time step size?
    (*logada_) << std::right << std::setw(16) << adareason_;

    // print L2-norms of the temporal discretization error indications
    (*logada_) << std::right << std::setw(16) << strnorm_ << std::right << std::setw(16)
               << strinnernorm_ << std::right << std::setw(16) << strfsinorm_ << std::right
               << std::setw(16) << flnorm_ << std::right << std::setw(16) << flinnernorm_
               << std::right << std::setw(16) << flfsinorm_;

    // print L-inf-norms of the temporal discretization error indications
    (*logada_) << std::right << std::setw(16) << strinfnorm_ << std::right << std::setw(16)
               << strinfinnernorm_ << std::right << std::setw(16) << strinffsinorm_ << std::right
               << std::setw(16) << flinfnorm_ << std::right << std::setw(16) << flinfinnernorm_
               << std::right << std::setw(16) << flinffsinorm_ << std::endl;
  }
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void FSI::Monolithic::print_adaptivity_summary() const
{
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
  {
    if (dt() != dt_past(1))  // only if time step size has changed
    {
      Core::IO::cout << "\n"
                     << "New time step size " << dt() << " is based on " << adareason_ << ".\n";
    }

    if (dtminused_)
    {
      Core::IO::cout << "Time step " << step() << " has been done with minimum time step size."
                     << " No further refinement possible. Go to next time step.\n";
    }

    if (not step_not_accepted())
    {
      Core::IO::cout << "Time step " << step() << " has been accepted after " << adaptstep_ - 1
                     << " repetitions."
                     << "\n";
    }
    else
    {
      Core::IO::cout << "Time step " << step() << " will be repeated with dt = " << dt() << ".\n";
    }
  }

  return;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void FSI::Monolithic::time_step_auxiliary()
{
  TEUCHOS_FUNC_TIME_MONITOR("FSI::Monolithic::time_step_auxiliary");

  // ---------------------------------------------------------------------------
  // Structure field
  // ---------------------------------------------------------------------------
  if (is_ada_structure())
  {
    std::dynamic_pointer_cast<Adapter::StructureFSITimIntAda>(structure_field())
        ->time_step_auxiliary();
  }
  // ---------------------------------------------------------------------------

  // ---------------------------------------------------------------------------
  // Fluid Field
  // ---------------------------------------------------------------------------
  if (is_ada_fluid())
  {
    fluid_field()->time_step_auxiliary();
  }
  // ---------------------------------------------------------------------------

  return;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void FSI::Monolithic::adapt_time_step_size()
{
  TEUCHOS_FUNC_TIME_MONITOR("FSI::Monolithic::adapt_time_step_size");

  // Increment counter for repetition of time steps
  adaptstep_++;

  // prepare new time step size by copying the current one
  double dtnew = dt();

  // ---------------------------------------------------------------------------
  // compute time step size suggestions based on structure field
  // ---------------------------------------------------------------------------
  if (is_ada_structure())
  {
    std::dynamic_pointer_cast<Adapter::StructureFSITimIntAda>(structure_field())
        ->indicate_error_norms(
            strnorm_, strfsinorm_, strinnernorm_, strinfnorm_, strinffsinorm_, strinfinnernorm_);

    // structure based time step size suggestions
    dtstr_ = std::dynamic_pointer_cast<Adapter::StructureFSITimIntAda>(structure_field())
                 ->calculate_dt(strnorm_);
    dtstrfsi_ = std::dynamic_pointer_cast<Adapter::StructureFSITimIntAda>(structure_field())
                    ->calculate_dt(strfsinorm_);
    dtstrinner_ = std::dynamic_pointer_cast<Adapter::StructureFSITimIntAda>(structure_field())
                      ->calculate_dt(strinnernorm_);
  }
  // ---------------------------------------------------------------------------

  // ---------------------------------------------------------------------------
  // compute time step size suggestions based on fluid field
  // ---------------------------------------------------------------------------
  if (is_ada_fluid())
  {
    fluid_field()->indicate_error_norms(
        flnorm_, flfsinorm_, flinnernorm_, flinfnorm_, flinffsinorm_, flinfinnernorm_);

    // error order
    const double order = fluid_field()->get_tim_ada_err_order();

    // calculate time step sizes resulting from errors in the fluid field
    dtfl_ = calculate_time_step_size(flnorm_, errtolfl_, order);
    dtflfsi_ = calculate_time_step_size(flfsinorm_, errtolfl_, order);
    dtflinner_ = calculate_time_step_size(flinnernorm_, errtolfl_, order);
  }
  // ---------------------------------------------------------------------------

  // ---------------------------------------------------------------------------
  // take care of possibly non-converged nonlinear solver
  // ---------------------------------------------------------------------------
  if (is_ada_solver())
  {
    switch (erroraction_)
    {
      case FSI::Monolithic::erroraction_halve_step:
      {
        dtnonlinsolver_ = std::max(0.5 * dt(), dtmin_);
        break;
      }
      case FSI::Monolithic::erroraction_revert_dt:
      {
        dtnonlinsolver_ = std::max(std::pow(0.95, (double)adaptstep_ - 1.0) * dt_past(0), dtmin_);
        break;
      }
      case FSI::Monolithic::erroraction_none:
      {
        dtnonlinsolver_ = std::min(std::max(1.1 * dt(), dtmin_), dtmax_);
        break;
      }
      default:
        break;
    }
  }
  else
  {
    dtnonlinsolver_ = dt();
  }
  // ---------------------------------------------------------------------------

  // Select new time step size
  dtnew = select_dt();

  // optional averaging of time step size in case of increasing time step size
  if (dtnew > dt_past(1))
  {
    dtnew *= avgweights_[0];
    for (unsigned int i = 1; i < avgweights_.size(); ++i) dtnew += avgweights_[i] * dt_past(-i + 1);

    if (dtnew > dtmax_)
      dtnew = dtmax_;
    else if (dtnew < dtmin_)
      dtnew = dtmin_;
  }

  if (dtnew <= dtmin_ and dt_past(1) <= dtmin_) dtminused_ = true;

  // Who is responsible for changing the time step size?
  determine_ada_reason(dtnew);

  // Check whether the step can be accepted, now
  accepted_ = set_accepted();

  // take care of possibly non-converged nonlinear solver
  if (is_ada_solver() and (erroraction_ == FSI::Monolithic::erroraction_halve_step or
                              erroraction_ == FSI::Monolithic::erroraction_revert_dt))
  {
    accepted_ = false;
  }

  // Finally, distribute new time step size to all participating fields/algorithms
  if (dtnew <= dt_past(1))  // time step size reduction is always done immediately
  {
    set_dt(dtnew);
  }
  else if (dtnew > dt_past(1))  // time step size increase may be delayed based on user's will
  {
    if (numincreasesteps_ <= 0)
    {
      // Finally, distribute new time step size to all participating fields/algorithms
      set_dt(dtnew);

      // reset
      numincreasesteps_ = Global::Problem::instance()
                              ->fsi_dynamic_params()
                              .sublist("TIMEADAPTIVITY")
                              .get<int>("NUMINCREASESTEPS");
    }
    else if (numincreasesteps_ > 0)
    {
      numincreasesteps_--;
    }
  }
  else
  {
    FOUR_C_THROW("Strange time step size!");
  }

  // ---------------------------------------------------------------------------
  return;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void FSI::Monolithic::determine_ada_reason(const double dt)
{
  const std::string oldreason = adareason_;

  if (is_ada_structure() and
      (dt == dtstr_ or dt == dtstrfsi_ or dt == dtstrinner_))  // structure field?
    adareason_ = "Structure";
  else if (is_ada_fluid() and (dt == dtfl_ or dt == dtflfsi_ or dt == dtflinner_))  // fluid field?
    adareason_ = "Fluid";
  else if (is_ada_solver() and dt == dtnonlinsolver_)  // Unconverged nonlinear solver?
    adareason_ = "Newton";

  // no change in time step size
  if (dt == dt_past(1)) adareason_ = oldreason;

  return;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
double FSI::Monolithic::calculate_time_step_size(
    const double errnorm, const double errtol, const double estorder) const
{
  //----------------------------------------------------------------------------
  // get some parameters first
  //----------------------------------------------------------------------------
  // FSI parameter list
  const Teuchos::ParameterList& fsidyn = Global::Problem::instance()->fsi_dynamic_params();

  // safety factor
  const double safetyfac = fsidyn.sublist("TIMEADAPTIVITY").get<double>("SAFETYFACTOR");

  // limiting factors for change of time step size
  const double facmax = fsidyn.sublist("TIMEADAPTIVITY").get<double>("SIZERATIOMAX");
  const double facmin = fsidyn.sublist("TIMEADAPTIVITY").get<double>("SIZERATIOMIN");

  // upper and lower bound of time step size
  const double dtmax = fsidyn.sublist("TIMEADAPTIVITY").get<double>("DTMAX");
  const double dtmin = fsidyn.sublist("TIMEADAPTIVITY").get<double>("DTMIN");
  //----------------------------------------------------------------------------

  //----------------------------------------------------------------------------
  // Calculate scaling factor
  //----------------------------------------------------------------------------
  // the optimal scaling factor
  double fac = 1.0;
  if (not(errnorm == 0.0))  // do not divide by zero
    fac = std::pow(errtol / errnorm, 1.0 / (estorder + 1.0));
  else  // max increase if error norm == 0
    fac = facmax / safetyfac;

  // limit by safety factor <= 1.0
  fac *= safetyfac;

  //----------------------------------------------------------------------------
  // Calculate new time step size
  //----------------------------------------------------------------------------
  // optimal new step size
  double dtnew = fac * dt();

  // redefine effective scaling factor to be dt*_{n}/dt_{n-1}, ie true optimal ratio
  fac = dtnew / dt_past(1);

  // limit effective scaling factor by maximum and minimum
  if (fac > facmax)
  {
    dtnew = facmax * dt_past(1);
  }
  else if (fac < facmin)
  {
    dtnew = facmin * dt_past(1);
  }

  // new step size subject to safety measurements
  if (dtnew > dtmax)
  {
    dtnew = dtmax;
  }
  else if (dtnew < dtmin)
  {
    dtnew = dtmin;
  }

  return dtnew;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void FSI::Monolithic::reset_step()
{
  if (is_ada_structure())
    std::dynamic_pointer_cast<Adapter::StructureFSITimIntAda>(structure_field())->reset_step();
  else
    structure_field()->reset_step();

  fluid_field()->reset_step();
  ale_field()->reset_step();
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void FSI::Monolithic::reset_time()
{
  // Fluid and ALE
  fluid_field()->reset_time(dt_past(1));
  ale_field()->reset_time(dt_past(1));

  // FSI routine
  set_time_step(time() - dt_past(1), step() - 1);
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void FSI::Monolithic::set_dt(const double dtnew)
{
  // single fields
  if (is_ada_structure())
    std::dynamic_pointer_cast<Adapter::StructureFSITimIntAda>(structure_field())->set_dt(dtnew);
  else
    structure_field()->set_dt(dtnew);

  fluid_field()->set_dt(dtnew);
  ale_field()->set_dt(dtnew);

  // FSI algorithm
  if (is_ada_structure() or is_ada_fluid() or is_ada_solver())
    dt_.set_step(1, dt());  // save step size of previous run of this time step for reset_time()

  Adapter::AlgorithmBase::set_dt(dtnew);

  return;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
double FSI::Monolithic::select_dt() const
{
  // First, take the new time step size based on error estimation
  double dtnew = select_dt_error_based();

  // Consider convergence of nonlinear solver if the user wants to.
  if (is_ada_solver())
  {
    // select time step size based on convergence of nonlinear solver
    if (erroraction_ != erroraction_none and erroraction_ != erroraction_continue)
    {
      dtnew = std::min(dtnew, dtnonlinsolver_);
    }
    else if (not(is_ada_structure() or is_ada_fluid()))
    {
      dtnew = dtnonlinsolver_;
    }
  }
  else
  {
    // no change in time step size based on convergence of nonlinear solver
  }

  return dtnew;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
bool FSI::Monolithic::check_if_dts_same()
{
  // get time step sizes from all fields
  const double dtfsi = dt();
  const double dtstruct = structure_field()->dt();
  const double dtfluid = fluid_field()->dt();
  const double dtale = ale_field()->dt();

  double dtstrada = -1.0;
  if (is_ada_structure())
    dtstrada = std::dynamic_pointer_cast<Adapter::StructureFSITimIntAda>(structure_field())->dt();

  // print time step sizes in all fields
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
  {
    std::cout << std::endl << "Time step sizes:" << std::endl;
    std::cout << "dt in FSI      : " << std::setprecision(16) << dtfsi << std::endl;
    std::cout << "dt in structure: " << std::setprecision(16) << dtstruct << std::endl;
    std::cout << "dt in fluid    : " << std::setprecision(16) << dtfluid << std::endl;
    std::cout << "dt in ale      : " << std::setprecision(16) << dtale << std::endl;

    if (is_ada_structure())
      std::cout << "dt in str_ada  : " << std::setprecision(16) << dtstrada << std::endl;
  }

  // check whether time step sizes are all the same
  const double tol = 1.0e-12;
  if (dtfsi - dtstruct < tol and dtfsi - dtfluid < tol and dtfsi - dtale < tol)
  {
    if (is_ada_structure() and dtfsi - dtstrada < tol)
      return true;
    else if (not is_ada_structure())
      return true;
    else
      return false;
  }
  else
  {
    FOUR_C_THROW("Time step sizes do not match among the fields.");
    return false;
  }
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
double FSI::Monolithic::dt_past(const int step) const { return dt_[step]; }

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void FSI::Monolithic::update_dt_past(const double dtnew) { dt_.update_steps(dtnew); }

FOUR_C_NAMESPACE_CLOSE
