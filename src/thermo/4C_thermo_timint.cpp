/*----------------------------------------------------------------------*/
/*! \file
\brief Time integration for spatially discretised thermal dynamics
\level 1
*/


/*----------------------------------------------------------------------*
 | headers                                                  bborn 08/09 |
 *----------------------------------------------------------------------*/
#include "4C_thermo_timint.hpp"

#include "4C_contact_nitsche_strategy_tsi.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_io_control.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_thermo_ele_action.hpp"
#include "4C_thermo_resulttest.hpp"
#include "4C_timestepping_mstep.hpp"
#include "4C_utils_function.hpp"

#include <Teuchos_TimeMonitor.hpp>

#include <iostream>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 | print thermal time logo                                   dano 08/09 |
 *----------------------------------------------------------------------*/
void Thermo::TimInt::logo()
{
  std::cout << "Welcome to Thermal Time Integration " << std::endl;
  std::cout << "      _______________________________" << std::endl;
  std::cout << "  ===(__|_|_37 degrees celsius__|_|__)" << std::endl;
  std::cout << std::endl;

}  // Logo()


/*----------------------------------------------------------------------*
 | constructor                                              bborn 08/09 |
 *----------------------------------------------------------------------*/
Thermo::TimInt::TimInt(const Teuchos::ParameterList& ioparams,
    const Teuchos::ParameterList& tdynparams, const Teuchos::ParameterList& xparams,
    Teuchos::RCP<Core::FE::Discretization> actdis, Teuchos::RCP<Core::LinAlg::Solver> solver,
    Teuchos::RCP<Core::IO::DiscretizationWriter> output)
    : discret_(actdis),
      myrank_(actdis->get_comm().MyPID()),
      solver_(solver),
      solveradapttol_(tdynparams.get<bool>("ADAPTCONV")),
      solveradaptolbetter_(tdynparams.get<double>("ADAPTCONV_BETTER")),
      dbcmaps_(Teuchos::make_rcp<Core::LinAlg::MapExtractor>()),
      output_(output),
      printlogo_(true),  // DON'T EVEN DARE TO SET THIS TO FALSE
      printscreen_(ioparams.get<int>("STDOUTEVRY")),
      printiter_(true),  // ADD INPUT PARAMETER
      writerestartevery_(tdynparams.get<int>("RESTARTEVRY")),
      writeglob_(ioparams.get<bool>("THERM_TEMPERATURE")),
      writeglobevery_(tdynparams.get<int>("RESULTSEVRY")),
      writeheatflux_(
          Teuchos::getIntegralValue<Inpar::Thermo::HeatFluxType>(ioparams, "THERM_HEATFLUX")),
      writetempgrad_(
          Teuchos::getIntegralValue<Inpar::Thermo::TempGradType>(ioparams, "THERM_TEMPGRAD")),
      writeenergyevery_(tdynparams.get<int>("RESEVRYERGY")),
      energyfile_(nullptr),
      calcerror_(Teuchos::getIntegralValue<Inpar::Thermo::CalcError>(tdynparams, "CALCERROR")),
      errorfunctno_(tdynparams.get<int>("CALCERRORFUNCNO")),
      time_(Teuchos::null),
      timen_(0.0),
      dt_(Teuchos::null),
      timemax_(tdynparams.get<double>("MAXTIME")),
      stepmax_(tdynparams.get<int>("NUMSTEP")),
      step_(0),
      stepn_(0),
      firstoutputofrun_(true),
      lumpcapa_(tdynparams.get<bool>("LUMPCAPA")),
      zeros_(Teuchos::null),
      temp_(Teuchos::null),
      rate_(Teuchos::null),
      tempn_(Teuchos::null),
      raten_(Teuchos::null),
      fifc_(Teuchos::null),
      tang_(Teuchos::null)
{
  // welcome user
  if ((printlogo_) and (myrank_ == 0))
  {
    logo();
  }

  // check wether discretisation has been completed
  if (not discret_->filled())
  {
    FOUR_C_THROW("Discretisation is not complete!");
  }

  // time state
  time_ = Teuchos::make_rcp<TimeStepping::TimIntMStep<double>>(0, 0, 0.0);
  // HERE SHOULD BE SOMETHING LIKE (tdynparams.get<double>("TIMEINIT"))
  dt_ = Teuchos::make_rcp<TimeStepping::TimIntMStep<double>>(
      0, 0, tdynparams.get<double>("TIMESTEP"));
  step_ = 0;
  timen_ = (*time_)[0] + (*dt_)[0];  // set target time to initial time plus step size
  stepn_ = step_ + 1;

  // output file for energy
  if ((writeenergyevery_ != 0) and (myrank_ == 0)) attach_energy_file();

  // a zero vector of full length
  zeros_ = Core::LinAlg::create_vector(*discret_->dof_row_map(), true);

  // Map containing Dirichlet DOFs
  {
    Teuchos::ParameterList p;
    p.set("total time", timen_);
    p.set<const Core::UTILS::FunctionManager*>(
        "function_manager", &Global::Problem::instance()->function_manager());
    discret_->evaluate_dirichlet(p, zeros_, Teuchos::null, Teuchos::null, Teuchos::null, dbcmaps_);
    zeros_->PutScalar(0.0);  // just in case of change
  }

  // temperatures T_{n}
  temp_ = Teuchos::make_rcp<TimeStepping::TimIntMStep<Core::LinAlg::Vector<double>>>(
      0, 0, discret_->dof_row_map(), true);
  // temperature rates R_{n}
  rate_ = Teuchos::make_rcp<TimeStepping::TimIntMStep<Core::LinAlg::Vector<double>>>(
      0, 0, discret_->dof_row_map(), true);

  // temperatures T_{n+1} at t_{n+1}
  tempn_ = Core::LinAlg::create_vector(*discret_->dof_row_map(), true);
  // temperature rates R_{n+1} at t_{n+1}
  raten_ = Core::LinAlg::create_vector(*discret_->dof_row_map(), true);

  // create empty interface force vector
  fifc_ = Core::LinAlg::create_vector(*discret_->dof_row_map(), true);

  // create empty matrix
  tang_ = Teuchos::make_rcp<Core::LinAlg::SparseMatrix>(*discret_->dof_row_map(), 81, true, true);
  // we condensed the capacity matrix out of the system

  // -------------------------------------------------------------------
  // set initial field
  // -------------------------------------------------------------------
  const int startfuncno = tdynparams.get<int>("INITFUNCNO");
  set_initial_field(
      Teuchos::getIntegralValue<Inpar::Thermo::InitialField>(tdynparams, "INITIALFIELD"),
      startfuncno);

  // stay with us
  return;

}  // cstr


/*----------------------------------------------------------------------*
 | equilibrate system at initial state                      bborn 08/09 |
 | and identify consistent temperature rate (only dynamic case)         |
 *----------------------------------------------------------------------*/
void Thermo::TimInt::determine_capa_consist_temp_rate()
{
  // temporary force vectors in this routine
  Teuchos::RCP<Core::LinAlg::Vector<double>> fext =
      Core::LinAlg::create_vector(*discret_->dof_row_map(), true);  //!< external force
  Teuchos::RCP<Core::LinAlg::Vector<double>> fint =
      Core::LinAlg::create_vector(*discret_->dof_row_map(), true);  //!< internal force

  // overwrite initial state vectors with DirichletBCs
  apply_dirichlet_bc((*time_)[0], (*temp_)(0), (*rate_)(0), false);

  // get external force
  apply_force_external((*time_)[0], (*temp_)(0), fext);
  // apply_force_external_conv is applied in the derived classes!

  // initialise matrices
  tang_->zero();
  //  capa_->Zero();

  // get initial internal force, tangent and capacity
  {
    // create the parameters for the discretization
    Teuchos::ParameterList p;
    // action for elements
    p.set<Thermo::Action>("action", Thermo::calc_thermo_fintcapa);
    // type of calling time integrator
    p.set<Inpar::Thermo::DynamicType>("time integrator", method_name());
    p.set<bool>("lump capa matrix", lumpcapa_);
    // other parameters that might be needed by the elements
    p.set("total time", (*time_)[0]);
    p.set("delta time", (*dt_)[0]);
    // set vector values needed by elements
    discret_->clear_state();
    // set_state(0,...) in case of multiple dofsets (e.g. TSI)
    discret_->set_state(0, "residual temperature", zeros_);
    discret_->set_state(0, "temperature", (*temp_)(0));

    // calculate the capacity matrix onto tang_, instead of buildung 2 matrices
    discret_->evaluate(p, Teuchos::null, tang_, fint, Teuchos::null, Teuchos::null);
    discret_->clear_state();
  }

  // finish capacity matrix
  //  capa_->Complete();

  // close tangent matrix
  tang_->complete();

  // calculate consistent initial temperature rates
  {
    // rhs corresponds to residual on the rhs
    // K . DT = - R_n+1 = - R_n - (fint_n+1 - fext_n+1)
    Teuchos::RCP<Core::LinAlg::Vector<double>> rhs =
        Core::LinAlg::create_vector(*discret_->dof_row_map(), true);
    rhs->Update(-1.0, *fint, 1.0, *fext, -1.0);
    // blank RHS on DBC DOFs
    dbcmaps_->insert_cond_vector(*dbcmaps_->extract_cond_vector(*zeros_), *rhs);
    Core::LinAlg::SolverParams solver_params;
    solver_params.refactor = true;
    solver_params.reset = true;
    solver_->solve(tang_->epetra_operator(), (*rate_)(0), rhs, solver_params);
  }

  // We need to reset the tangent matrix because its graph (topology)
  // is not finished yet in case of constraints and possibly other side
  // effects (basically managers).
  // BUT: in case of explicit time integration, the conductivity matrix
  // is stored in tang_ which is needed throughout the simulation
  if (method_name() != Inpar::Thermo::dyna_expleuler) tang_->reset();

  // leave this hell
  return;

}  // determine_capa_consist_temp_rate()


/*----------------------------------------------------------------------*
 | evaluate Dirichlet BC at t_{n+1}                         bborn 06/08 |
 *----------------------------------------------------------------------*/
void Thermo::TimInt::apply_dirichlet_bc(const double time,
    Teuchos::RCP<Core::LinAlg::Vector<double>> temp,
    Teuchos::RCP<Core::LinAlg::Vector<double>> rate, bool recreatemap)
{
  // apply DBCs
  // needed parameters
  Teuchos::ParameterList p;
  p.set("total time", time);  // target time
  p.set<const Core::UTILS::FunctionManager*>(
      "function_manager", &Global::Problem::instance()->function_manager());

  // predicted Dirichlet values
  // \c temp then also holds prescribed new Dirichlet temperatures
  discret_->clear_state();
  if (recreatemap)
  {
    discret_->evaluate_dirichlet(p, temp, rate, Teuchos::null, Teuchos::null, dbcmaps_);
  }
  else
  {
    discret_->evaluate_dirichlet(p, temp, rate, Teuchos::null, Teuchos::null, Teuchos::null);
  }
  discret_->clear_state();

  // ciao
  return;

}  // apply_dirichlet_bc()


/*----------------------------------------------------------------------*
 | update time and step counter                            bborn 06/08 |
 *----------------------------------------------------------------------*/
void Thermo::TimInt::update_step_time()
{
  // update time and step
  time_->update_steps(timen_);  // t_{n} := t_{n+1}, etc
  step_ = stepn_;               // n := n+1

  timen_ += (*dt_)[0];
  stepn_ += 1;

  // new deal
  return;

}  // UpdateStepTime()


/*----------------------------------------------------------------------*
 | reset configuration after time step                      bborn 06/08 |
 *----------------------------------------------------------------------*/
void Thermo::TimInt::reset_step()
{
  // reset state vectors
  tempn_->Update(1.0, (*temp_)[0], 0.0);
  raten_->Update(1.0, (*rate_)[0], 0.0);

  // reset anything that needs to be reset at the element level
  {
    // create the parameters for the discretization
    Teuchos::ParameterList p;
    p.set<Thermo::Action>("action", Thermo::calc_thermo_reset_istep);
    p.set("total time", time());
    p.set("delta time", dt());
    // go to elements
    discret_->evaluate(
        p, Teuchos::null, Teuchos::null, Teuchos::null, Teuchos::null, Teuchos::null);
    discret_->clear_state();
  }

  // I am gone
  return;

}  // reset_step()


/*----------------------------------------------------------------------*
 | read and set restart values                              bborn 06/08 |
 *----------------------------------------------------------------------*/
void Thermo::TimInt::read_restart(const int step)
{
  Core::IO::DiscretizationReader reader(
      discret_, Global::Problem::instance()->input_control_file(), step);
  if (step != reader.read_int("step")) FOUR_C_THROW("Time step on file not equal to given step");

  step_ = step;
  stepn_ = step_ + 1;
  time_ = Teuchos::make_rcp<TimeStepping::TimIntMStep<double>>(0, 0, reader.read_double("time"));
  timen_ = (*time_)[0] + (*dt_)[0];

  read_restart_state();
  read_restart_force();

}  // read_restart()


/*----------------------------------------------------------------------*
 | read and set restart state                               bborn 06/08 |
 *----------------------------------------------------------------------*/
void Thermo::TimInt::read_restart_state()
{
  Core::IO::DiscretizationReader reader(
      discret_, Global::Problem::instance()->input_control_file(), step_);
  reader.read_vector(tempn_, "temperature");
  temp_->update_steps(*tempn_);
  reader.read_vector(raten_, "rate");
  rate_->update_steps(*raten_);
  reader.read_history_data(step_);
  return;

}  // ReadRestartState()


/*----------------------------------------------------------------------*
 | output to file                                           mwgee 03/07 |
 *----------------------------------------------------------------------*/
void Thermo::TimInt::output_step(bool forced_writerestart)
{
  // special treatment is necessary when restart is forced
  if (forced_writerestart)
  {
    // restart has already been written or simulation has just started
    if ((writerestartevery_ and (step_ % writerestartevery_ == 0)) or
        step_ == Global::Problem::instance()->restart())
      return;
    // if state already exists, add restart information
    if (writeglobevery_ and (step_ % writeglobevery_ == 0))
    {
      add_restart_to_output_state();
      return;
    }
  }

  // this flag is passed along subroutines and prevents
  // repeated initialising of output writer, printing of
  // state vectors, or similar
  bool datawritten = false;

  // output restart (try this first)
  // write restart step
  if ((writerestartevery_ and (step_ % writerestartevery_ == 0)) or forced_writerestart)
  {
    output_restart(datawritten);
  }

  // output results (not necessary if restart in same step)
  if (writeglob_ and writeglobevery_ and (step_ % writeglobevery_ == 0) and (not datawritten))
  {
    output_state(datawritten);
  }

  // output heatflux & tempgrad
  if (writeglobevery_ and (step_ % writeglobevery_ == 0) and
      ((writeheatflux_ != Inpar::Thermo::heatflux_none) or
          (writetempgrad_ != Inpar::Thermo::tempgrad_none)))
  {
    output_heatflux_tempgrad(datawritten);
  }

  // output energy
  if (writeenergyevery_ and (step_ % writeenergyevery_ == 0))
  {
    output_energy();
  }

  // what's next?
  return;

}  // OutputStep()


/*----------------------------------------------------------------------*
 | write restart                                            mwgee 03/07 |
 *----------------------------------------------------------------------*/
void Thermo::TimInt::output_restart(bool& datawritten)
{
  // Yes, we are going to write...
  datawritten = true;

  // write restart output, please
  output_->write_mesh(step_, (*time_)[0]);
  output_->new_step(step_, (*time_)[0]);
  output_->write_vector("temperature", (*temp_)(0));
  output_->write_vector("rate", (*rate_)(0));
  // write all force vectors which are later read in restart
  write_restart_force(output_);
  // owner of elements is just written once because it does not change during simulation (so far)
  output_->write_element_data(firstoutputofrun_);
  firstoutputofrun_ = false;

  // info dedicated to user's eyes staring at standard out
  if ((myrank_ == 0) and printscreen_ and (step_old() % printscreen_ == 0))
  {
    printf("====== Restart written in step %d\n", step_);
    // print a beautiful line made exactly of 80 dashes
    printf(
        "--------------------------------------------------------------"
        "------------------\n");
    fflush(stdout);
  }

}  // output_restart()


/*----------------------------------------------------------------------*
 | output temperature,temperature rate                      bborn 06/08 |
 | originally by mwgee 03/07                                            |
 *----------------------------------------------------------------------*/
void Thermo::TimInt::output_state(bool& datawritten)
{
  // Yes, we are going to write...
  datawritten = true;

  // write now
  output_->new_step(step_, (*time_)[0]);
  output_->write_vector("temperature", (*temp_)(0));
  output_->write_vector("rate", (*rate_)(0));
  // owner of elements is just written once because it does not change during simulation (so far)
  output_->write_element_data(firstoutputofrun_);
  firstoutputofrun_ = false;

  // leave for good
  return;

}  // output_state()


/*----------------------------------------------------------------------*
 | add restart information to OutputStatewrite restart      ghamm 10/13 |
 *----------------------------------------------------------------------*/
void Thermo::TimInt::add_restart_to_output_state()
{
  // write all force vectors which are later read in restart
  write_restart_force(output_);

  // finally add the missing mesh information, order is important here
  output_->write_mesh(step_, (*time_)[0]);

  // info dedicated to user's eyes staring at standard out
  if ((myrank_ == 0) and printscreen_ and (step_old() % printscreen_ == 0))
  {
    printf("====== Restart written in step %d\n", step_);
    // print a beautiful line made exactly of 80 dashes
    printf(
        "--------------------------------------------------------------"
        "------------------\n");
    fflush(stdout);
  }
}  // add_restart_to_output_state()


/*----------------------------------------------------------------------*
 | heatflux calculation and output                          bborn 06/08 |
 | originally by lw                                                     |
 *----------------------------------------------------------------------*/
void Thermo::TimInt::output_heatflux_tempgrad(bool& datawritten)
{
  // create the parameters for the discretization
  Teuchos::ParameterList p;
  // action for elements
  p.set<Thermo::Action>("action", Thermo::calc_thermo_heatflux);
  // other parameters that might be needed by the elements
  p.set("total time", (*time_)[0]);
  p.set("delta time", (*dt_)[0]);

  Teuchos::RCP<std::vector<char>> heatfluxdata = Teuchos::make_rcp<std::vector<char>>();
  p.set("heatflux", heatfluxdata);
  p.set<Inpar::Thermo::HeatFluxType>("ioheatflux", writeheatflux_);

  Teuchos::RCP<std::vector<char>> tempgraddata = Teuchos::make_rcp<std::vector<char>>();
  p.set("tempgrad", tempgraddata);
  p.set<Inpar::Thermo::TempGradType>("iotempgrad", writetempgrad_);

  // set vector values needed by elements
  discret_->clear_state();
  // set_state(0,...) in case of multiple dofsets (e.g. TSI)
  discret_->set_state(0, "residual temperature", zeros_);
  discret_->set_state(0, "temperature", (*temp_)(0));

  discret_->evaluate(p, Teuchos::null, Teuchos::null, Teuchos::null, Teuchos::null, Teuchos::null);
  discret_->clear_state();

  // Make new step
  if (not datawritten)
  {
    output_->new_step(step_, (*time_)[0]);
  }
  datawritten = true;

  // write heatflux
  if (writeheatflux_ != Inpar::Thermo::heatflux_none)
  {
    std::string heatfluxtext = "";
    if (writeheatflux_ == Inpar::Thermo::heatflux_current)
    {
      heatfluxtext = "gauss_current_heatfluxes_xyz";
    }
    else if (writeheatflux_ == Inpar::Thermo::heatflux_initial)
    {
      heatfluxtext = "gauss_initial_heatfluxes_xyz";
    }
    else
    {
      FOUR_C_THROW("requested heatflux type not supported");
    }
    output_->write_vector(heatfluxtext, *heatfluxdata, *(discret_->element_col_map()));
  }

  // write temperature gradient
  if (writetempgrad_ != Inpar::Thermo::tempgrad_none)
  {
    std::string tempgradtext = "";
    if (writetempgrad_ == Inpar::Thermo::tempgrad_current)
    {
      tempgradtext = "gauss_current_tempgrad_xyz";
    }
    else if (writetempgrad_ == Inpar::Thermo::tempgrad_initial)
    {
      tempgradtext = "gauss_initial_tempgrad_xyz";
    }
    else
    {
      FOUR_C_THROW("requested tempgrad type not supported");
    }
    output_->write_vector(tempgradtext, *tempgraddata, *(discret_->element_col_map()));
  }

  // leave me alone
  return;

}  // output_heatflux_tempgrad()


/*----------------------------------------------------------------------*
 | output system energies                                   bborn 06/08 |
 *----------------------------------------------------------------------*/
void Thermo::TimInt::output_energy()
{
  // internal/tempgrad energy
  double intergy = 0.0;  // total internal energy
  {
    Teuchos::ParameterList p;
    // other parameters needed by the elements
    p.set<Thermo::Action>("action", Thermo::calc_thermo_energy);

    // set vector values needed by elements
    discret_->clear_state();
    // set_state(0,...) in case of multiple dofsets (e.g. TSI)
    discret_->set_state(0, "temperature", (*temp_)(0));
    // get energies
    Teuchos::RCP<Core::LinAlg::SerialDenseVector> energies =
        Teuchos::make_rcp<Core::LinAlg::SerialDenseVector>(1);
    discret_->evaluate_scalars(p, energies);
    discret_->clear_state();
    intergy = (*energies)(0);
  }

  // the output
  if (myrank_ == 0)
  {
    *energyfile_ << " " << std::setw(9) << step_ << std::scientific << std::setprecision(16) << " "
                 << (*time_)[0] << " " << intergy << std::endl;
  }

  // in God we trust
  return;

}  // output_energy()


/*----------------------------------------------------------------------*
 | thermal result test                                       dano 01/12 |
 *----------------------------------------------------------------------*/
Teuchos::RCP<Core::UTILS::ResultTest> Thermo::TimInt::create_field_test()
{
  return Teuchos::make_rcp<Thermo::ResultTest>(*this);

}  // CreateFieldTest()


/*----------------------------------------------------------------------*
 | evaluate external forces at t_{n+1}                      bborn 06/08 |
 *----------------------------------------------------------------------*/
void Thermo::TimInt::apply_force_external(const double time,  //!< evaluation time
    const Teuchos::RCP<Core::LinAlg::Vector<double>> temp,    //!< temperature state
    Teuchos::RCP<Core::LinAlg::Vector<double>>& fext          //!< external force
)
{
  Teuchos::ParameterList p;
  // action for elements
  const Thermo::Action action = Thermo::calc_thermo_fext;
  p.set<Thermo::Action>("action", action);
  // type of calling time integrator
  p.set<Inpar::Thermo::DynamicType>("time integrator", method_name());
  // other parameters needed by the elements
  p.set("total time", time);

  // set vector values needed by elements
  discret_->clear_state();
  // set_state(0,...) in case of multiple dofsets (e.g. TSI)
  discret_->set_state(0, "temperature", temp);
  // get load vector
  discret_->evaluate_neumann(p, *fext);
  discret_->clear_state();

  // go away
  return;

}  // apply_force_external()


/*----------------------------------------------------------------------*
 | evaluate convection boundary conditions at t_{n+1}        dano 01/11 |
 *----------------------------------------------------------------------*/
void Thermo::TimInt::apply_force_external_conv(Teuchos::ParameterList& p,
    const double time,                                       //!< evaluation time
    const Teuchos::RCP<Core::LinAlg::Vector<double>> tempn,  //!< temperature state T_n
    const Teuchos::RCP<Core::LinAlg::Vector<double>> temp,   //!< temperature state T_n+1
    Teuchos::RCP<Core::LinAlg::Vector<double>>& fext,        //!< external force
    Teuchos::RCP<Core::LinAlg::SparseOperator> tang          //!< tangent at time n+1
)
{
  // for heat convection von Neumann boundary conditions, i.e. q_c^, the
  // calculation depends on the deformation, i.e. differentiation between
  // geo_linear and geo_nonlinear is required
  // - geo_linear:
  //   - use CalculateFindCondCapa(), contribution to linearisation for k_TT
  // geo_nonlinear:
  //   - use CalculateNlnFindCondCapa() considering deformation d_{n+1}
  //   - contribution due to linearisation for k_TT AND k_Td

  // action for elements
  const Thermo::BoundaryAction boundaryAction = Thermo::calc_thermo_fextconvection;
  p.set<Thermo::BoundaryAction>("action", boundaryAction);
  // type of calling time integrator
  p.set<Inpar::Thermo::DynamicType>("time integrator", method_name());
  // other parameters needed by the elements
  p.set("total time", time);

  // set vector values needed by elements
  discret_->clear_state();
  discret_->set_state(0, "old temperature", tempn);  // T_n (*temp_)(0)
  discret_->set_state(0, "temperature", temp);       // T_{n+1} tempn_

  // get load vector
  // use general version of evaluate_condition()
  std::string condstring("ThermoConvections");
  discret_->evaluate_condition(
      p, tang, Teuchos::null, fext, Teuchos::null, Teuchos::null, condstring);
  discret_->clear_state();

  // go away
  return;

}  // apply_force_external_conv()


/*----------------------------------------------------------------------*
 | evaluate ordinary internal force, its tangent at state   bborn 06/08 |
 *----------------------------------------------------------------------*/
void Thermo::TimInt::apply_force_tang_internal(
    Teuchos::ParameterList& p, const double time, const double dt,
    const Teuchos::RCP<Core::LinAlg::Vector<double>> temp,   //!< temperature state
    const Teuchos::RCP<Core::LinAlg::Vector<double>> tempi,  //!< residual temperature
    Teuchos::RCP<Core::LinAlg::Vector<double>> fint,         //!< internal force
    Teuchos::RCP<Core::LinAlg::SparseMatrix> tang            //!< tangent matrix
)
{
  // type of calling time integrator
  p.set<Inpar::Thermo::DynamicType>("time integrator", method_name());
  // action for elements
  const Thermo::Action action = Thermo::calc_thermo_fintcond;
  p.set<Thermo::Action>("action", action);
  // other parameters that might be needed by the elements
  p.set("total time", time);
  p.set("delta time", dt);
  // set vector values needed by elements
  discret_->clear_state();
  // set_state(0,...) in case of multiple dofsets (e.g. TSI)
  discret_->set_state(0, "residual temperature", tempi);
  discret_->set_state(0, "temperature", temp);

  discret_->evaluate(p, tang, Teuchos::null, fint, Teuchos::null, Teuchos::null);

  // apply contact terms
  if (contact_strategy_nitsche_ != Teuchos::null)
  {
    if (fint->Update(
            1., *contact_strategy_nitsche_->get_rhs_block_ptr(CONTACT::VecBlockType::temp), 1.))
      FOUR_C_THROW("update failed");
    if (contact_params_interface_->get_coupling_scheme() ==
        Inpar::CONTACT::CouplingScheme::monolithic)
    {
      tang->un_complete();
      tang->add(*contact_strategy_nitsche_->get_matrix_block_ptr(CONTACT::MatBlockType::temp_temp),
          false, p.get<double>("timefac"), 1.);
      tang->complete();
    }
  }

  discret_->clear_state();

  // that's it
  return;

}  // apply_force_tang_internal()


/*----------------------------------------------------------------------*
 | evaluate ordinary internal force, its tangent at state   bborn 10/09 |
 | overloaded function specified for ost time integration               |
 *----------------------------------------------------------------------*/
void Thermo::TimInt::apply_force_tang_internal(
    Teuchos::ParameterList& p, const double time, const double dt,
    const Teuchos::RCP<Core::LinAlg::Vector<double>> temp,   //!< temperature state
    const Teuchos::RCP<Core::LinAlg::Vector<double>> tempi,  //!< residual temperature
    Teuchos::RCP<Core::LinAlg::Vector<double>> fcap,         //!< capacity force
    Teuchos::RCP<Core::LinAlg::Vector<double>> fint,         //!< internal force
    Teuchos::RCP<Core::LinAlg::SparseMatrix> tang            //!< tangent matrix
)
{
  // type of calling time integrator
  p.set<Inpar::Thermo::DynamicType>("time integrator", method_name());
  // action for elements
  const Thermo::Action action = Thermo::calc_thermo_finttang;
  p.set<Thermo::Action>("action", action);
  // other parameters that might be needed by the elements
  p.set("total time", time);
  p.set("delta time", dt);
  // set vector values needed by elements
  discret_->clear_state();
  // set_state(0,...) in case of multiple dofsets (e.g. TSI)
  discret_->set_state(0, "residual temperature", tempi);
  discret_->set_state(0, "temperature", temp);
  // required for linearization of T-dependent capacity
  discret_->set_state(0, "last temperature", (*temp_)(0));

  // in case of genalpha extract midpoint temperature rate R_{n+alpha_m}
  // extract it after ClearState() is called.
  if (method_name() == Inpar::Thermo::dyna_genalpha)
  {
    Teuchos::RCP<const Core::LinAlg::Vector<double>> ratem =
        p.get<Teuchos::RCP<const Core::LinAlg::Vector<double>>>("mid-temprate");
    if (ratem != Teuchos::null) discret_->set_state(0, "mid-temprate", ratem);
  }

  // call the element evaluate()
  discret_->evaluate(p, tang, Teuchos::null, fint, Teuchos::null, fcap);

  // apply contact terms
  if (contact_strategy_nitsche_ != Teuchos::null)
  {
    fint->Update(
        1., *contact_strategy_nitsche_->get_rhs_block_ptr(CONTACT::VecBlockType::temp), 1.);
    if (contact_params_interface_->get_coupling_scheme() ==
        Inpar::CONTACT::CouplingScheme::monolithic)
    {
      tang->un_complete();
      tang->add(*contact_strategy_nitsche_->get_matrix_block_ptr(CONTACT::MatBlockType::temp_temp),
          false, p.get<double>("timefac"), 1.);
      tang->complete();
    }
  }

  discret_->clear_state();

  // that's it
  return;

}  // apply_force_tang_internal()


/*----------------------------------------------------------------------*
 | evaluate ordinary internal force                         bborn 06/08 |
 *----------------------------------------------------------------------*/
void Thermo::TimInt::apply_force_internal(
    Teuchos::ParameterList& p, const double time, const double dt,
    const Teuchos::RCP<Core::LinAlg::Vector<double>> temp,   //!< temperature state
    const Teuchos::RCP<Core::LinAlg::Vector<double>> tempi,  //!< incremental temperature
    Teuchos::RCP<Core::LinAlg::Vector<double>> fint          //!< internal force
)
{
  // type of calling time integrator
  p.set<Inpar::Thermo::DynamicType>("time integrator", method_name());
  // action for elements
  Thermo::Action action = Thermo::calc_thermo_fint;
  p.set<Thermo::Action>("action", action);
  // other parameters that might be needed by the elements
  p.set("total time", time);
  p.set("delta time", dt);
  // set vector values needed by elements
  discret_->clear_state();
  // set_state(0,...) in case of multiple dofsets (e.g. TSI)
  discret_->set_state(0, "residual temperature", tempi);
  discret_->set_state(0, "temperature", temp);

  // call the element evaluate()
  discret_->evaluate(p, Teuchos::null, Teuchos::null, fint, Teuchos::null, Teuchos::null);
  discret_->clear_state();

  // apply contact terms
  if (contact_strategy_nitsche_ != Teuchos::null)
    fint->Update(
        1., *contact_strategy_nitsche_->get_rhs_block_ptr(CONTACT::VecBlockType::temp), 1.);

  // where the fun starts
  return;

}  // apply_force_tang_internal()


/*----------------------------------------------------------------------*
 | set initial field for temperature (according to ScaTra)   dano 06/10 |
 *----------------------------------------------------------------------*/
void Thermo::TimInt::set_initial_field(
    const Inpar::Thermo::InitialField init, const int startfuncno)
{
  switch (init)
  {
    case Inpar::Thermo::initfield_zero_field:
    {
      // extract temperature vector at time t_n (temp_ contains various vectors of
      // old(er) temperatures and is of type TimIntMStep<Core::LinAlg::Vector<double>>)
      (*temp_)(0)->PutScalar(0.0);
      tempn_->PutScalar(0.0);
      break;
    }  // initfield_zero_field

    case Inpar::Thermo::initfield_field_by_function:
    {
      const Epetra_Map* dofrowmap = discret_->dof_row_map();

      // loop all nodes on the processor
      for (int lnodeid = 0; lnodeid < discret_->num_my_row_nodes(); lnodeid++)
      {
        // get the processor local node
        Core::Nodes::Node* lnode = discret_->l_row_node(lnodeid);
        // the set of degrees of freedom associated with the node
        std::vector<int> nodedofset = discret_->dof(0, lnode);

        int numdofs = nodedofset.size();
        for (int k = 0; k < numdofs; ++k)
        {
          const int dofgid = nodedofset[k];
          int doflid = dofrowmap->LID(dofgid);
          // evaluate component k of spatial function
          double initialval =
              Global::Problem::instance()
                  ->function_by_id<Core::UTILS::FunctionOfSpaceTime>(startfuncno - 1)
                  .evaluate(lnode->x().data(), 0.0, k);
          // extract temperature vector at time t_n (temp_ contains various vectors of
          // old(er) temperatures and is of type TimIntMStep<Core::LinAlg::Vector<double>>)
          int err1 = (*temp_)(0)->ReplaceMyValues(1, &initialval, &doflid);
          if (err1 != 0) FOUR_C_THROW("dof not on proc");
          // initialise also the solution vector. These values are a pretty good
          // guess for the solution after the first time step (much better than
          // starting with a zero vector)
          int err2 = tempn_->ReplaceMyValues(1, &initialval, &doflid);
          if (err2 != 0) FOUR_C_THROW("dof not on proc");
        }  // numdofs
      }
      break;
    }  // initfield_field_by_function

    case Inpar::Thermo::initfield_field_by_condition:
    {
      std::vector<int> localdofs;
      localdofs.push_back(0);
      discret_->evaluate_initial_field(
          Global::Problem::instance()->function_manager(), "Temperature", *(*temp_)(0), localdofs);
      discret_->evaluate_initial_field(
          Global::Problem::instance()->function_manager(), "Temperature", *tempn_, localdofs);

      break;
    }  // initfield_field_by_condition

    default:
      FOUR_C_THROW("Unknown option for initial field: %d", init);
      break;
  }  // switch(init)

  // and back
  return;

}  // SetInitialField()


/*----------------------------------------------------------------------*
 | apply interface loads to the thermo field                ghamm 12/10 |
 *----------------------------------------------------------------------*/
void Thermo::TimInt::set_force_interface(Teuchos::RCP<Core::LinAlg::Vector<double>> ithermoload)
{
  fifc_->Update(1.0, *ithermoload, 0.0);
}  // SetForceInterface()

/*-----------------------------------------------------------------------------*
 *   evaluate error compared to analytical solution                vuong 03/15 |
 *----------------------------------------------------------------------------*/
Teuchos::RCP<std::vector<double>> Thermo::TimInt::evaluate_error_compared_to_analytical_sol()
{
  switch (calcerror_)
  {
    case Inpar::Thermo::no_error_calculation:
    {
      // do nothing --- no analytical solution available
      return Teuchos::null;
      break;
    }
    case Inpar::Thermo::calcerror_byfunct:
    {
      // std::vector containing
      // [0]: relative L2 temperature error
      // [1]: relative H1 temperature error
      Teuchos::RCP<std::vector<double>> relerror = Teuchos::make_rcp<std::vector<double>>(2);

      // create the parameters for the discretization
      Teuchos::ParameterList eleparams;

      // action for elements
      eleparams.set("total time", timen_);
      eleparams.set<Thermo::Action>("action", Thermo::calc_thermo_error);
      eleparams.set<Inpar::Thermo::CalcError>("calculate error", calcerror_);
      eleparams.set<int>("error function number", errorfunctno_);

      discret_->set_state("temperature", tempn_);

      // get (squared) error values
      // 0: delta temperature for L2-error norm
      // 1: delta temperature for H1-error norm
      // 2: analytical temperature for L2 norm
      // 3: analytical temperature for H1 norm
      Teuchos::RCP<Core::LinAlg::SerialDenseVector> errors =
          Teuchos::make_rcp<Core::LinAlg::SerialDenseVector>(4);

      // vector for output
      Teuchos::RCP<Epetra_MultiVector> normvec =
          Teuchos::make_rcp<Epetra_MultiVector>(*discret_->element_row_map(), 7);

      // call loop over elements (assemble nothing)
      discret_->evaluate_scalars(eleparams, errors);
      discret_->evaluate_scalars(eleparams, *normvec);
      discret_->clear_state();

      (*relerror)[0] = sqrt((*errors)[0]) / sqrt((*errors)[2]);
      (*relerror)[1] = sqrt((*errors)[1]) / sqrt((*errors)[3]);

      if (myrank_ == 0)
      {
        {
          std::cout.precision(8);
          std::cout << std::endl
                    << "---- error norm for analytical solution type " << calcerror_
                    << " ----------" << std::endl;
          std::cout << "| relative L_2 temperature error norm:     " << (*relerror)[0] << std::endl;
          std::cout << "| absolute H_1 temperature error norm:     " << (*relerror)[1] << std::endl;
          std::cout << "--------------------------------------------------------------------"
                    << std::endl
                    << std::endl;
          std::cout << "H1 temperature scaling  " << sqrt((*errors)[3]) << std::endl;
        }

        // print last error in a seperate file

        // append error of the last time step to the error file
        if ((step_ == stepmax_) or ((*time_)[0] == timemax_))  // write results to file
        {
          std::ostringstream temp;
          const std::string simulation =
              Global::Problem::instance()->output_control_file()->file_name();
          const std::string fname = simulation + "_thermo.relerror";

          std::ofstream f;
          f.open(fname.c_str(), std::fstream::ate | std::fstream::app);
          f << "#| " << simulation << "\n";
          f << "#| Step | Time | rel. L2-error temperature  |  rel. H1-error temperature  |\n";
          f << step_ << " " << (*time_)[0] << " " << (*relerror)[0] << " " << (*relerror)[1]
            << "\n";
          f.flush();
          f.close();
        }

        const std::string simulation =
            Global::Problem::instance()->output_control_file()->file_name();
        const std::string fname = simulation + "_thermo_time.relerror";

        if (step_ == 1)
        {
          std::ofstream f;
          f.open(fname.c_str());
          f << "#| Step | Time | rel. L2-error temperature  |  rel. H1-error temperature  |\n";
          f << std::setprecision(10) << step_ << " " << std::setw(1) << std::setprecision(5)
            << (*time_)[0] << std::setw(1) << std::setprecision(6) << " " << (*relerror)[0]
            << std::setw(1) << std::setprecision(6) << " " << (*relerror)[1] << "\n";

          f.flush();
          f.close();
        }
        else
        {
          std::ofstream f;
          f.open(fname.c_str(), std::fstream::ate | std::fstream::app);
          f << std::setprecision(10) << step_ << " " << std::setw(3) << std::setprecision(5)
            << (*time_)[0] << std::setw(1) << std::setprecision(6) << " " << (*relerror)[0]
            << std::setw(1) << std::setprecision(6) << " " << (*relerror)[1] << "\n";

          f.flush();
          f.close();
        }
      }

      return relerror;
    }
    default:
      FOUR_C_THROW("unknown type of error calculation!");
      return Teuchos::null;
  }
}  // end evaluate_error_compared_to_analytical_sol
/*----------------------------------------------------------------------*/

FOUR_C_NAMESPACE_CLOSE
