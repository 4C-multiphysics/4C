// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_thermo_timint.hpp"

#include "4C_fem_discretization_nullspace.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_io_control.hpp"
#include "4C_io_discretization_visualization_writer_mesh.hpp"
#include "4C_io_visualization_parameters.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_thermo_ele_action.hpp"
#include "4C_thermo_resulttest.hpp"
#include "4C_timestepping_mstep.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_function.hpp"
#include "4C_utils_parameter_list.hpp"
#include "4C_utils_shared_ptr_from_ref.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>
#include <Teuchos_TimeMonitor.hpp>

#include <iostream>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Thermo::TimInt::TimInt(const Teuchos::ParameterList& ioparams,
    const Teuchos::ParameterList& tdynparams, const Teuchos::ParameterList& xparams,
    std::shared_ptr<Core::FE::Discretization> actdis, std::shared_ptr<Core::LinAlg::Solver> solver,
    std::shared_ptr<Core::IO::DiscretizationWriter> output)
    : discret_(actdis),
      solver_(solver),
      solveradapttol_(tdynparams.get<bool>("ADAPTCONV")),
      solveradaptolbetter_(tdynparams.get<double>("ADAPTCONV_BETTER")),
      dbcmaps_(std::make_shared<Core::LinAlg::MapExtractor>()),
      output_(output),
      printscreen_(ioparams.get<int>("STDOUTEVERY")),
      writerestartevery_(tdynparams.get<int>("RESTARTEVERY")),
      writeglobevery_(tdynparams.get<int>("RESULTSEVERY")),
      writeheatflux_(Teuchos::getIntegralValue<Thermo::HeatFluxType>(ioparams, "THERM_HEATFLUX")),
      writetempgrad_(Teuchos::getIntegralValue<Thermo::TempGradType>(ioparams, "THERM_TEMPGRAD")),
      calcerror_(Teuchos::getIntegralValue<Thermo::CalcError>(tdynparams, "CALCERROR")),
      errorfunctno_(tdynparams.get<int>("CALCERRORFUNCNO")),
      timen_(0.0),
      timemax_(tdynparams.get<double>("MAXTIME")),
      stepmax_(tdynparams.get<int>("NUMSTEP")),
      step_(0),
      stepn_(0),
      lumpcapa_(tdynparams.get<bool>("LUMPCAPA")),
      zeros_(nullptr),
      tempn_(nullptr),
      raten_(nullptr),
      tang_(nullptr)
{
  if (Core::Communication::my_mpi_rank(discret_->get_comm()) == 0)
  {
    std::cout << "Welcome to Thermal Time Integration " << std::endl;
    std::cout << "      _______________________________" << std::endl;
    std::cout << "  ===(__|_|_37 degrees celsius__|_|__)" << std::endl;
    std::cout << std::endl;
  }

  // check whether discretisation has been completed
  if (not discret_->filled())
  {
    FOUR_C_THROW("Discretisation is not complete!");
  }

  // time state
  dt_ = TimeStepping::TimIntMStep(0, 0, tdynparams.get<double>("TIMESTEP"));
  if (int step = Global::Problem::instance()->restart(); step > 0)
  {
    Core::IO::DiscretizationReader reader(
        *discret_, Global::Problem::instance()->input_control_file(), step);
    if (step != reader.read_int("step")) FOUR_C_THROW("Time step on file not equal to given step");

    step_ = step;
    time_ = TimeStepping::TimIntMStep<double>(0, 0, reader.read_double("time"));
    timen_ = time_[0] + dt_[0];
  }
  else
  {
    step_ = 0;
    time_ = TimeStepping::TimIntMStep<double>(0, 0, 0.0);
    timen_ = time_[0] + dt_[0];
  }
  stepn_ = step_ + 1;

  // a zero vector of full length
  zeros_ = std::make_shared<Core::LinAlg::Vector<double>>(*discret_->dof_row_map(), true);

  // Map containing Dirichlet DOFs
  {
    Teuchos::ParameterList p;
    p.set("total time", timen_);
    p.set<const Core::Utils::FunctionManager*>(
        "function_manager", &Global::Problem::instance()->function_manager());
    discret_->evaluate_dirichlet(p, zeros_, nullptr, nullptr, nullptr, dbcmaps_);
    zeros_->put_scalar(0.0);  // just in case of change
  }

  // temperatures T_{n}
  temp_ =
      TimeStepping::TimIntMStep<Core::LinAlg::Vector<double>>(0, 0, discret_->dof_row_map(), true);
  // temperature rates R_{n}
  rate_ =
      TimeStepping::TimIntMStep<Core::LinAlg::Vector<double>>(0, 0, discret_->dof_row_map(), true);

  // temperatures T_{n+1} at t_{n+1}
  tempn_ = std::make_shared<Core::LinAlg::Vector<double>>(*discret_->dof_row_map(), true);
  // temperature rates R_{n+1} at t_{n+1}
  raten_ = std::make_shared<Core::LinAlg::Vector<double>>(*discret_->dof_row_map(), true);

  // create empty matrix
  tang_ = std::make_shared<Core::LinAlg::SparseMatrix>(*discret_->dof_row_map(), 81, true, true);
  // we condensed the capacity matrix out of the system

  // -------------------------------------------------------------------
  // set initial field
  // -------------------------------------------------------------------
  const int startfuncno = tdynparams.get<int>("INITFUNCNO");
  set_initial_field(
      Teuchos::getIntegralValue<Thermo::InitialField>(tdynparams, "INITIALFIELD"), startfuncno);

  // check for special thermo vtk output which is to be handled by an own writer object
  {
    const Teuchos::ParameterList thermo_vtk_runtime_output_list(
        tdynparams.sublist("RUNTIME VTK OUTPUT"));

    auto vtk_output_thermo = thermo_vtk_runtime_output_list.get<bool>("OUTPUT_THERMO");

    if (vtk_output_thermo)
    {
      bool output_temperature_state = thermo_vtk_runtime_output_list.get<bool>("TEMPERATURE");
      bool output_temperature_rate_state =
          thermo_vtk_runtime_output_list.get<bool>("TEMPERATURE_RATE");
      bool output_conductivity_state = thermo_vtk_runtime_output_list.get<bool>("CONDUCTIVITY");
      bool output_element_owner = thermo_vtk_runtime_output_list.get<bool>("ELEMENT_OWNER");
      bool output_element_gid = thermo_vtk_runtime_output_list.get<bool>("ELEMENT_GID");
      bool output_node_gid = thermo_vtk_runtime_output_list.get<bool>("NODE_GID");

      runtime_vtk_params_ = {.output_temperature_state = output_temperature_state,
          .output_temperature_rate_state = output_temperature_rate_state,
          .output_conductivity_state = output_conductivity_state,
          .output_element_owner = output_element_owner,
          .output_element_gid = output_element_gid,
          .output_node_gid = output_node_gid};

      runtime_vtk_writer_ = Core::IO::DiscretizationVisualizationWriterMesh(
          discret_, Core::IO::visualization_parameters_factory(
                        Global::Problem::instance()->io_params().sublist("RUNTIME VTK OUTPUT"),
                        *Global::Problem::instance()->output_control_file(), time_[0]));
    }
  }

  // check for special thermo csv output which is to be handled by an own writer object
  {
    const Teuchos::ParameterList thermo_csv_runtime_output_list(
        tdynparams.sublist("RUNTIME CSV OUTPUT"));

    auto csv_output_thermo = thermo_csv_runtime_output_list.get<bool>("OUTPUT_THERMO");

    if (csv_output_thermo)
    {
      runtime_csv_writer_.emplace(Core::Communication::my_mpi_rank(discret_->get_comm()),
          *Global::Problem::instance()->output_control_file(), "thermo");

      bool output_energy_state = thermo_csv_runtime_output_list.get<bool>("ENERGY");

      if (output_energy_state)
      {
        runtime_csv_writer_->register_data_vector("energy", 1, 16);
      }

      runtime_csv_params_ = {output_energy_state};
    }
  }
}


/*----------------------------------------------------------------------*
 | equilibrate system at initial state                      bborn 08/09 |
 | and identify consistent temperature rate (only dynamic case)         |
 *----------------------------------------------------------------------*/
void Thermo::TimInt::determine_capa_consist_temp_rate()
{
  // temporary force vectors in this routine
  std::shared_ptr<Core::LinAlg::Vector<double>> fext =
      std::make_shared<Core::LinAlg::Vector<double>>(
          *discret_->dof_row_map(), true);  //!< external force
  std::shared_ptr<Core::LinAlg::Vector<double>> fint =
      std::make_shared<Core::LinAlg::Vector<double>>(
          *discret_->dof_row_map(), true);  //!< internal force

  // overwrite initial state vectors with DirichletBCs
  apply_dirichlet_bc(time_[0], temp_(0), rate_(0), false);

  // get external force
  apply_force_external(time_[0], temp_(0), *fext);
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
    p.set<Thermo::DynamicType>("time integrator", method_name());
    p.set<bool>("lump capa matrix", lumpcapa_);
    // other parameters that might be needed by the elements
    p.set("total time", time_[0]);
    p.set("delta time", dt_[0]);
    // set vector values needed by elements
    discret_->clear_state();
    // set_state(0,...) in case of multiple dofsets (e.g. TSI)
    discret_->set_state(0, "residual temperature", *zeros_);
    discret_->set_state(0, "temperature", *temp_(0));

    // calculate the capacity matrix onto tang_, instead of buildung 2 matrices
    discret_->evaluate(p, nullptr, tang_, fint, nullptr, nullptr);
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
    std::shared_ptr<Core::LinAlg::Vector<double>> rhs =
        std::make_shared<Core::LinAlg::Vector<double>>(*discret_->dof_row_map(), true);
    rhs->update(-1.0, *fint, 1.0, *fext, -1.0);
    // blank RHS on DBC DOFs
    dbcmaps_->insert_cond_vector(*dbcmaps_->extract_cond_vector(*zeros_), *rhs);

    compute_null_space_if_necessary(*discret_, solver_->params());

    Core::LinAlg::SolverParams solver_params;
    solver_params.refactor = true;
    solver_params.reset = true;
    solver_->solve(tang_, rate_(0), rhs, solver_params);
  }

  // We need to reset the tangent matrix because its graph (topology)
  // is not finished yet in case of constraints and possibly other side
  // effects (basically managers).
  tang_->reset();
}  // determine_capa_consist_temp_rate()


/*----------------------------------------------------------------------*
 | evaluate Dirichlet BC at t_{n+1}                         bborn 06/08 |
 *----------------------------------------------------------------------*/
void Thermo::TimInt::apply_dirichlet_bc(const double time,
    std::shared_ptr<Core::LinAlg::Vector<double>> temp,
    std::shared_ptr<Core::LinAlg::Vector<double>> rate, bool recreatemap)
{
  // apply DBCs
  // needed parameters
  Teuchos::ParameterList p;
  p.set("total time", time);  // target time
  p.set<const Core::Utils::FunctionManager*>(
      "function_manager", &Global::Problem::instance()->function_manager());

  // predicted Dirichlet values
  // \c temp then also holds prescribed new Dirichlet temperatures
  discret_->clear_state();
  if (recreatemap)
  {
    discret_->evaluate_dirichlet(p, temp, rate, nullptr, nullptr, dbcmaps_);
  }
  else
  {
    discret_->evaluate_dirichlet(p, temp, rate, nullptr, nullptr, nullptr);
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
  time_.update_steps(timen_);  // t_{n} := t_{n+1}, etc
  step_ = stepn_;              // n := n+1

  timen_ += dt_[0];
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
  tempn_->update(1.0, temp_[0], 0.0);
  raten_->update(1.0, rate_[0], 0.0);

  // reset anything that needs to be reset at the element level
  {
    // create the parameters for the discretization
    Teuchos::ParameterList p;
    p.set<Thermo::Action>("action", Thermo::calc_thermo_reset_istep);
    p.set("total time", time());
    p.set("delta time", dt());
    // go to elements
    discret_->evaluate(p, nullptr, nullptr, nullptr, nullptr, nullptr);
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
  // step and time is already set in initialisation of time integrator

  read_restart_state();
  read_restart_force();

}  // read_restart()


/*----------------------------------------------------------------------*
 | read and set restart state                               bborn 06/08 |
 *----------------------------------------------------------------------*/
void Thermo::TimInt::read_restart_state()
{
  Core::IO::DiscretizationReader reader(
      *discret_, Global::Problem::instance()->input_control_file(), step_);
  reader.read_vector(tempn_, "temperature");
  temp_.update_steps(*tempn_);
  reader.read_vector(raten_, "rate");
  rate_.update_steps(*raten_);
  reader.read_history_data(step_);
  return;

}  // ReadRestartState()


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Thermo::TimInt::write_runtime_output()
{
  // write vtk runtime output
  if (runtime_vtk_writer_.has_value())
  {
    runtime_vtk_writer_->reset();

    if (runtime_vtk_params_.output_temperature_state)
    {
      runtime_vtk_writer_->append_result_data_vector_with_context(
          *tempn_, Core::IO::OutputEntity::node, {"temperature"});
    }

    if (runtime_vtk_params_.output_temperature_rate_state)
    {
      runtime_vtk_writer_->append_result_data_vector_with_context(
          *raten_, Core::IO::OutputEntity::node, {"temperature_rate"});
    }

    if (runtime_vtk_params_.output_conductivity_state)
    {
      std::vector<std::optional<std::string>> context(conductivity_->num_vectors(), "conductivity");
      runtime_vtk_writer_->append_result_data_vector_with_context(
          *conductivity_, Core::IO::OutputEntity::node, context);
    }

    if (runtime_vtk_params_.output_element_owner)
      runtime_vtk_writer_->append_element_owner("element_owner");

    if (runtime_vtk_params_.output_element_gid)
      runtime_vtk_writer_->append_element_gid("element_gid");

    if (runtime_vtk_params_.output_node_gid) runtime_vtk_writer_->append_node_gid("node_gid");

    // finalize everything and write all required files to filesystem
    runtime_vtk_writer_->write_to_disk(time_[0], step_);
  }

  // write csv runtime output
  if (runtime_csv_writer_.has_value())
  {
    if (runtime_csv_params_.output_energy_state)
    {
      double energy = 0.0;

      Teuchos::ParameterList p;
      p.set<Thermo::Action>("action", Thermo::calc_thermo_energy);

      discret_->clear_state();
      discret_->set_state(0, "temperature", *temp_(0));

      Core::LinAlg::SerialDenseVector energies(1);
      discret_->evaluate_scalars(p, energies);
      discret_->clear_state();
      energy = (energies)(0);

      std::map<std::string, std::vector<double>> output_energy;
      output_energy["energy"] = {energy};
      runtime_csv_writer_->write_data_to_file(time_[0], step_, output_energy);
    }
  }
}


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

  // write runtime output
  if (writeglobevery_ > 0 and step_ % writeglobevery_ == 0)
  {
    write_runtime_output();
  }

  // output heatflux & tempgrad
  if (writeglobevery_ and (step_ % writeglobevery_ == 0) and
      ((writeheatflux_ != Thermo::heatflux_none) or (writetempgrad_ != Thermo::tempgrad_none)))
  {
    output_heatflux_tempgrad(datawritten);
  }
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Thermo::TimInt::output_restart(bool& datawritten)
{
  datawritten = true;

  output_->write_mesh(step_, time_[0]);
  output_->new_step(step_, time_[0]);
  output_->write_vector("temperature", temp_(0));
  output_->write_vector("rate", rate_(0));
  write_restart_force(output_);

  if ((Core::Communication::my_mpi_rank(discret_->get_comm()) == 0) and printscreen_ and
      (step_old() % printscreen_ == 0))
  {
    Core::IO::cout << "====== Restart written in step " << step_ << "\n";
    Core::IO::cout
        << "--------------------------------------------------------------------------------"
        << "\n";
  }
}

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
  p.set("total time", time_[0]);
  p.set("delta time", dt_[0]);

  std::shared_ptr<std::vector<char>> heatfluxdata = std::make_shared<std::vector<char>>();
  p.set("heatflux", heatfluxdata);
  p.set<Thermo::HeatFluxType>("ioheatflux", writeheatflux_);

  std::shared_ptr<std::vector<char>> tempgraddata = std::make_shared<std::vector<char>>();
  p.set("tempgrad", tempgraddata);
  p.set<Thermo::TempGradType>("iotempgrad", writetempgrad_);

  // set vector values needed by elements
  discret_->clear_state();
  // set_state(0,...) in case of multiple dofsets (e.g. TSI)
  discret_->set_state(0, "residual temperature", *zeros_);
  discret_->set_state(0, "temperature", *temp_(0));

  Core::LinAlg::Vector<double> heatflux(*discret_->dof_row_map(), true);

  discret_->evaluate(p, nullptr, nullptr, nullptr, nullptr, nullptr);
  discret_->clear_state();

  // Make new step
  if (not datawritten)
  {
    output_->new_step(step_, time_[0]);
  }
  datawritten = true;

  // write heatflux
  if (writeheatflux_ != Thermo::heatflux_none)
  {
    std::string heatfluxtext = "";
    if (writeheatflux_ == Thermo::heatflux_current)
    {
      heatfluxtext = "gauss_current_heatfluxes_xyz";
    }
    else if (writeheatflux_ == Thermo::heatflux_initial)
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
  if (writetempgrad_ != Thermo::tempgrad_none)
  {
    std::string tempgradtext = "";
    if (writetempgrad_ == Thermo::tempgrad_current)
    {
      tempgradtext = "gauss_current_tempgrad_xyz";
    }
    else if (writetempgrad_ == Thermo::tempgrad_initial)
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
 | thermal result test                                       dano 01/12 |
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::Utils::ResultTest> Thermo::TimInt::create_field_test()
{
  return std::make_shared<Thermo::ResultTest>(*this);
}  // CreateFieldTest()


/*----------------------------------------------------------------------*
 | evaluate external forces at t_{n+1}                      bborn 06/08 |
 *----------------------------------------------------------------------*/
void Thermo::TimInt::apply_force_external(const double time,   //!< evaluation time
    const std::shared_ptr<Core::LinAlg::Vector<double>> temp,  //!< temperature state
    Core::LinAlg::Vector<double>& fext                         //!< external force
)
{
  Teuchos::ParameterList p;
  // action for elements
  const Thermo::Action action = Thermo::calc_thermo_fext;
  p.set<Thermo::Action>("action", action);
  // type of calling time integrator
  p.set<Thermo::DynamicType>("time integrator", method_name());
  // other parameters needed by the elements
  p.set("total time", time);

  // set vector values needed by elements
  discret_->clear_state();
  // set_state(0,...) in case of multiple dofsets (e.g. TSI)
  discret_->set_state(0, "temperature", *temp);
  // get load vector
  discret_->evaluate_neumann(p, fext);
  discret_->clear_state();

  // go away
  return;

}  // apply_force_external()


/*----------------------------------------------------------------------*
 | evaluate convection boundary conditions at t_{n+1}        dano 01/11 |
 *----------------------------------------------------------------------*/
void Thermo::TimInt::apply_force_external_conv(Teuchos::ParameterList& p,
    const double time,                                          //!< evaluation time
    const std::shared_ptr<Core::LinAlg::Vector<double>> tempn,  //!< temperature state T_n
    const std::shared_ptr<Core::LinAlg::Vector<double>> temp,   //!< temperature state T_n+1
    std::shared_ptr<Core::LinAlg::Vector<double>>& fext,        //!< external force
    std::shared_ptr<Core::LinAlg::SparseOperator> tang          //!< tangent at time n+1
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
  p.set<Thermo::DynamicType>("time integrator", method_name());
  // other parameters needed by the elements
  p.set("total time", time);

  // set vector values needed by elements
  discret_->clear_state();
  discret_->set_state(0, "old temperature", *tempn);  // T_n (*temp_)(0)
  discret_->set_state(0, "temperature", *temp);       // T_{n+1} tempn_

  // get load vector
  // use general version of evaluate_condition()
  std::string condstring("ThermoConvections");
  discret_->evaluate_condition(p, tang, nullptr, fext, nullptr, nullptr, condstring);
  discret_->clear_state();

  // go away
  return;

}  // apply_force_external_conv()


/*----------------------------------------------------------------------*
 | evaluate ordinary internal force, its tangent at state   bborn 06/08 |
 *----------------------------------------------------------------------*/
void Thermo::TimInt::apply_force_tang_internal(
    Teuchos::ParameterList& p, const double time, const double dt,
    const std::shared_ptr<Core::LinAlg::Vector<double>> temp,   //!< temperature state
    const std::shared_ptr<Core::LinAlg::Vector<double>> tempi,  //!< residual temperature
    std::shared_ptr<Core::LinAlg::Vector<double>> fint,         //!< internal force
    std::shared_ptr<Core::LinAlg::SparseMatrix> tang            //!< tangent matrix
)
{
  // type of calling time integrator
  p.set<Thermo::DynamicType>("time integrator", method_name());
  // action for elements
  const Thermo::Action action = Thermo::calc_thermo_fintcond;
  p.set<Thermo::Action>("action", action);
  // other parameters that might be needed by the elements
  p.set("total time", time);
  p.set("delta time", dt);
  // set vector values needed by elements
  discret_->clear_state();
  // set_state(0,...) in case of multiple dofsets (e.g. TSI)
  discret_->set_state(0, "residual temperature", *tempi);
  discret_->set_state(0, "temperature", *temp);

  discret_->evaluate(p, tang, nullptr, fint, nullptr, nullptr);

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
    const std::shared_ptr<Core::LinAlg::Vector<double>> temp,   //!< temperature state
    const std::shared_ptr<Core::LinAlg::Vector<double>> tempi,  //!< residual temperature
    std::shared_ptr<Core::LinAlg::Vector<double>> fcap,         //!< capacity force
    std::shared_ptr<Core::LinAlg::Vector<double>> fint,         //!< internal force
    std::shared_ptr<Core::LinAlg::SparseMatrix> tang            //!< tangent matrix
)
{
  // type of calling time integrator
  p.set<Thermo::DynamicType>("time integrator", method_name());
  // action for elements
  const Thermo::Action action = Thermo::calc_thermo_finttang;
  p.set<Thermo::Action>("action", action);
  // other parameters that might be needed by the elements
  p.set("total time", time);
  p.set("delta time", dt);
  // set vector values needed by elements
  discret_->clear_state();
  // set_state(0,...) in case of multiple dofsets (e.g. TSI)
  discret_->set_state(0, "residual temperature", *tempi);
  discret_->set_state(0, "temperature", *temp);
  // required for linearization of T-dependent capacity
  discret_->set_state(0, "last temperature", *temp_(0));

  // in case of genalpha extract midpoint temperature rate R_{n+alpha_m}
  // extract it after ClearState() is called.
  if (method_name() == Thermo::DynamicType::GenAlpha)
  {
    std::shared_ptr<const Core::LinAlg::Vector<double>> ratem =
        p.get<std::shared_ptr<const Core::LinAlg::Vector<double>>>("mid-temprate");
    if (ratem != nullptr) discret_->set_state(0, "mid-temprate", *ratem);
  }

  // call the element evaluate()
  discret_->evaluate(p, tang, nullptr, fint, nullptr, fcap);

  discret_->clear_state();

  // that's it
  return;

}  // apply_force_tang_internal()


/*----------------------------------------------------------------------*
 | evaluate ordinary internal force                         bborn 06/08 |
 *----------------------------------------------------------------------*/
void Thermo::TimInt::apply_force_internal(
    Teuchos::ParameterList& p, const double time, const double dt,
    const std::shared_ptr<Core::LinAlg::Vector<double>> temp,   //!< temperature state
    const std::shared_ptr<Core::LinAlg::Vector<double>> tempi,  //!< incremental temperature
    std::shared_ptr<Core::LinAlg::Vector<double>> fint          //!< internal force
)
{
  // type of calling time integrator
  p.set<Thermo::DynamicType>("time integrator", method_name());
  // action for elements
  Thermo::Action action = Thermo::calc_thermo_fint;
  p.set<Thermo::Action>("action", action);
  // other parameters that might be needed by the elements
  p.set("total time", time);
  p.set("delta time", dt);
  // set vector values needed by elements
  discret_->clear_state();
  // set_state(0,...) in case of multiple dofsets (e.g. TSI)
  discret_->set_state(0, "residual temperature", *tempi);
  discret_->set_state(0, "temperature", *temp);

  // call the element evaluate()
  discret_->evaluate(p, nullptr, nullptr, fint, nullptr, nullptr);
  discret_->clear_state();
}  // apply_force_tang_internal()


/*----------------------------------------------------------------------*
 | set initial field for temperature (according to ScaTra)   dano 06/10 |
 *----------------------------------------------------------------------*/
void Thermo::TimInt::set_initial_field(const Thermo::InitialField init, const int startfuncno)
{
  switch (init)
  {
    case Thermo::initfield_zero_field:
    {
      // extract temperature vector at time t_n (temp_ contains various vectors of
      // old(er) temperatures and is of type TimIntMStep<Core::LinAlg::Vector<double>>)
      temp_(0)->put_scalar(0.0);
      tempn_->put_scalar(0.0);
      break;
    }  // initfield_zero_field

    case Thermo::initfield_field_by_function:
    {
      const Core::LinAlg::Map* dofrowmap = discret_->dof_row_map();

      for (auto lnode : discret_->my_row_node_range())
      {
        // the set of degrees of freedom associated with the node
        std::vector<int> nodedofset = discret_->dof(0, lnode);

        int numdofs = nodedofset.size();
        for (int k = 0; k < numdofs; ++k)
        {
          const int dofgid = nodedofset[k];
          int doflid = dofrowmap->lid(dofgid);
          // evaluate component k of spatial function
          double initialval = Global::Problem::instance()
                                  ->function_by_id<Core::Utils::FunctionOfSpaceTime>(startfuncno)
                                  .evaluate(lnode.x(), 0.0, k);
          // extract temperature vector at time t_n (temp_ contains various vectors of
          // old(er) temperatures and is of type TimIntMStep<Core::LinAlg::Vector<double>>)
          temp_(0)->replace_local_value(doflid, initialval);
          // initialise also the solution vector. These values are a pretty good
          // guess for the solution after the first time step (much better than
          // starting with a zero vector)
          tempn_->replace_local_value(doflid, initialval);
        }  // numdofs
      }
      break;
    }  // initfield_field_by_function

    case Thermo::initfield_field_by_condition:
    {
      std::vector<int> localdofs;
      localdofs.push_back(0);
      discret_->evaluate_initial_field(
          Global::Problem::instance()->function_manager(), "Temperature", *temp_(0), localdofs);
      discret_->evaluate_initial_field(
          Global::Problem::instance()->function_manager(), "Temperature", *tempn_, localdofs);

      break;
    }  // initfield_field_by_condition

    default:
      FOUR_C_THROW("Unknown option for initial field: {}", init);
      break;
  }  // switch(init)

  // and back
  return;

}  // SetInitialField()

/*-----------------------------------------------------------------------------*
 *   evaluate error compared to analytical solution                vuong 03/15 |
 *----------------------------------------------------------------------------*/
std::shared_ptr<std::vector<double>> Thermo::TimInt::evaluate_error_compared_to_analytical_sol()
{
  switch (calcerror_)
  {
    case Thermo::no_error_calculation:
    {
      // do nothing --- no analytical solution available
      return nullptr;
      break;
    }
    case Thermo::calcerror_byfunct:
    {
      // std::vector containing
      // [0]: relative L2 temperature error
      // [1]: relative H1 temperature error
      std::shared_ptr<std::vector<double>> relerror = std::make_shared<std::vector<double>>(2);

      // create the parameters for the discretization
      Teuchos::ParameterList eleparams;

      // action for elements
      eleparams.set("total time", timen_);
      eleparams.set<Thermo::Action>("action", Thermo::calc_thermo_error);
      eleparams.set<Thermo::CalcError>("calculate error", calcerror_);
      eleparams.set<int>("error function number", errorfunctno_);

      discret_->set_state("temperature", *tempn_);

      // get (squared) error values
      // 0: delta temperature for L2-error norm
      // 1: delta temperature for H1-error norm
      // 2: analytical temperature for L2 norm
      // 3: analytical temperature for H1 norm
      Core::LinAlg::SerialDenseVector errors(4);

      // vector for output
      Core::LinAlg::MultiVector<double> normvec(*discret_->element_row_map(), 7);

      // call loop over elements (assemble nothing)
      discret_->evaluate_scalars(eleparams, errors);
      discret_->evaluate_scalars(eleparams, normvec);
      discret_->clear_state();

      (*relerror)[0] = sqrt((errors)[0]) / sqrt((errors)[2]);
      (*relerror)[1] = sqrt((errors)[1]) / sqrt((errors)[3]);

      if (Core::Communication::my_mpi_rank(discret_->get_comm()) == 0)
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
          std::cout << "H1 temperature scaling  " << sqrt((errors)[3]) << std::endl;
        }

        // print last error in a separate file

        // append error of the last time step to the error file
        if ((step_ == stepmax_) or (time_[0] == timemax_))  // write results to file
        {
          std::ostringstream temp;
          const std::string simulation =
              Global::Problem::instance()->output_control_file()->file_name();
          const std::string fname = simulation + "_thermo.relerror";

          std::ofstream f;
          f.open(fname.c_str(), std::fstream::ate | std::fstream::app);
          f << "#| " << simulation << "\n";
          f << "#| Step | Time | rel. L2-error temperature  |  rel. H1-error temperature  |\n";
          f << step_ << " " << time_[0] << " " << (*relerror)[0] << " " << (*relerror)[1] << "\n";
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
            << time_[0] << std::setw(1) << std::setprecision(6) << " " << (*relerror)[0]
            << std::setw(1) << std::setprecision(6) << " " << (*relerror)[1] << "\n";

          f.flush();
          f.close();
        }
        else
        {
          std::ofstream f;
          f.open(fname.c_str(), std::fstream::ate | std::fstream::app);
          f << std::setprecision(10) << step_ << " " << std::setw(3) << std::setprecision(5)
            << time_[0] << std::setw(1) << std::setprecision(6) << " " << (*relerror)[0]
            << std::setw(1) << std::setprecision(6) << " " << (*relerror)[1] << "\n";

          f.flush();
          f.close();
        }
      }

      return relerror;
    }
    default:
      FOUR_C_THROW("unknown type of error calculation!");
      return nullptr;
  }
}  // end evaluate_error_compared_to_analytical_sol
/*----------------------------------------------------------------------*/

FOUR_C_NAMESPACE_CLOSE
