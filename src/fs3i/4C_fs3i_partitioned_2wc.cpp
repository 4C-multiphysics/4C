// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fs3i_partitioned_2wc.hpp"

#include "4C_adapter_fld_fluid_fsi.hpp"
#include "4C_adapter_str_fsiwrapper.hpp"
#include "4C_adapter_structure_scatra_ele.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fluid_timint_loma.hpp"
#include "4C_fs3i_problem_access.hpp"
#include "4C_fsi_monolithic.hpp"
#include "4C_global_data.hpp"
#include "4C_scatra_algorithm.hpp"
#include "4C_scatra_timint_loma.hpp"

FOUR_C_NAMESPACE_OPEN

namespace
{
  const Teuchos::ParameterList& fs3i_dynamic_params_from_problem()
  {
    return FS3I::Utils::fs3i_dynamic_params_from_problem();
  }
}  // namespace

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
FS3I::PartFS3I2Wc::PartFS3I2Wc(MPI_Comm comm)
    : PartFS3I(comm),
      itmax_(fs3i_dynamic_params_from_problem().sublist("PARTITIONED").get<int>("ITEMAX")),
      ittol_(fs3i_dynamic_params_from_problem().sublist("PARTITIONED").get<double>("CONVTOL")),
      consthermpress_(fs3i_dynamic_params_from_problem().get<std::string>("CONSTHERMPRESS"))
{
  // constructor is supposed to stay empty
  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FS3I::PartFS3I2Wc::init()
{
  // call init() in base class
  FS3I::PartFS3I::init();

  return;
}

void FS3I::PartFS3I2Wc::validate_structure_scatra_clone_configuration(
    const Core::FE::Discretization& structdis, const Core::FE::Discretization& structscatradis,
    const std::map<std::pair<std::string, std::string>, std::map<int, int>>& clonefieldmatmap) const
{
  bool requires_loma_mapping = false;
  for (int i = 0; i < structdis.num_my_col_elements(); ++i)
  {
    if (Adapter::get_sca_tra_impl_type(structdis.l_col_element(i)) == ScaTra::impltype_loma)
    {
      requires_loma_mapping = true;
      break;
    }
  }

  if (!requires_loma_mapping) return;

  const std::pair<std::string, std::string> key(structdis.name(), structscatradis.name());
  if (clonefieldmatmap.contains(key)) return;

  FOUR_C_THROW(
      "Structure-based low-Mach scalar transport requires a CLONING MATERIAL MAP entry from "
      "'{}' to '{}'.",
      structdis.name(), structscatradis.name());
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FS3I::PartFS3I2Wc::setup()
{
  // call setup() in base class
  FS3I::PartFS3I::setup();

  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FS3I::PartFS3I2Wc::timeloop()
{
  check_is_init();
  check_is_setup();

  initial_calculations();

  // prepare time loop
  fsi_->prepare_timeloop();
  set_fsi_solution();

  // calculate initial time derivative, when restart was done from a part. FSI simulation
  const auto* problem = FS3I::Utils::problem_from_instance();
  if (problem->restart() and problem->f_s3_i_dynamic_params().get<bool>("RESTART_FROM_PART_FSI"))
  {
    scatravec_[0]->scatra_field()->prepare_first_time_step();
    scatravec_[1]->scatra_field()->prepare_first_time_step();
  }

  // output of initial state
  constexpr bool force_prepare = true;
  fsi_->prepare_output(force_prepare);
  fsi_->output();
  scatra_output();

  while (not_finished())
  {
    increment_time_and_step();

    prepare_time_step();

    outer_loop();

    time_update_and_output();
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FS3I::PartFS3I2Wc::initial_calculations()
{
  // set initial fluid velocity field for evaluation of initial scalar
  // time derivative in fluid-based scalar transport
  scatravec_[0]->scatra_field()->set_convective_velocity(*fsi_->fluid_field()->velnp());

  // set initial value of thermodynamic pressure in fluid-based scalar
  // transport
  // for constant thermodynamic pressure in low-Mach-number flow and
  // for temperature-dependent water, thermodynamic pressure is set
  // to a constant here and never touched again
  if ((fsi_->fluid_field()->physical_type() == Inpar::FLUID::tempdepwater) &&
      (consthermpress_ != "Yes"))
    FOUR_C_THROW(
        "Constant thermodynamic pressure required if TFSI algorithm is used with "
        "temperature-dependent water!");
  std::dynamic_pointer_cast<ScaTra::ScaTraTimIntLoma>(scatravec_[0]->scatra_field())
      ->set_initial_therm_pressure();

  // mass conservation: compute initial mass (initial time deriv. assumed zero)
  if (consthermpress_ == "No_mass")
    std::dynamic_pointer_cast<ScaTra::ScaTraTimIntLoma>(scatravec_[0]->scatra_field())
        ->compute_initial_mass();

  // set initial scalar field and thermodynamic pressure for evaluation of
  // Neumann boundary conditions in fluid at beginning of first time step
  fsi_->fluid_field()->set_scalar_fields(scatravec_[0]->scatra_field()->phinp(),
      std::dynamic_pointer_cast<ScaTra::ScaTraTimIntLoma>(scatravec_[0]->scatra_field())
          ->therm_press_np(),
      nullptr, scatravec_[0]->scatra_field()->discretization());

  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FS3I::PartFS3I2Wc::prepare_time_step()
{
  check_is_init();
  check_is_setup();

  // prepare time step for both fluid- and structure-based scatra field
  for (unsigned i = 0; i < scatravec_.size(); ++i)
  {
    std::shared_ptr<Adapter::ScaTraBaseAlgorithm> scatra = scatravec_[i];
    scatra->scatra_field()->prepare_time_step();
  }

  // predict thermodynamic pressure and time derivative
  // (if not constant or based on mass conservation)
  if (consthermpress_ == "No_energy")
    std::dynamic_pointer_cast<ScaTra::ScaTraTimIntLoma>(scatravec_[0]->scatra_field())
        ->predict_therm_pressure();

  // prepare time step for fluid, structure and ALE fields
  fsi_->prepare_time_step();

  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FS3I::PartFS3I2Wc::outer_loop()
{
  // set iteration number and stop criterion
  int itnum = 0;
  bool stopnonliniter = false;

  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
  {
    std::cout << "\n****************************************\n          OUTER ITERATION "
                 "LOOP\n****************************************\n";

    printf("TIME: %11.4E/%11.4E  DT = %11.4E  %s  STEP = %4d/%4d\n",
        scatravec_[0]->scatra_field()->time(), timemax_, dt_,
        scatravec_[0]->scatra_field()->method_title().c_str(),
        scatravec_[0]->scatra_field()->step(), numstep_);
  }

  // set mesh displacement and velocity fields
  // TO DO: temporally consistent transfer of velocity and other fields,
  // for the time being, zero velocity field from structure
  set_fsi_solution();

  // initially solve coupled scalar transport equation system
  // (values for intermediate time steps were calculated at the end of prepare_time_step)
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
    std::cout << "\n****************************************\n        SCALAR TRANSPORT "
                 "SOLVER\n****************************************\n";
  scatra_evaluate_solve_iter_update();

  while (stopnonliniter == false)
  {
    itnum++;

    // in case of non-constant thermodynamic pressure: compute
    // (either based on energy conservation or based on mass conservation)
    if (consthermpress_ == "No_energy")
      std::dynamic_pointer_cast<ScaTra::ScaTraTimIntLoma>(scatravec_[0]->scatra_field())
          ->compute_therm_pressure();
    else if (consthermpress_ == "No_mass")
      std::dynamic_pointer_cast<ScaTra::ScaTraTimIntLoma>(scatravec_[0]->scatra_field())
          ->compute_therm_pressure_from_mass_cons();

    // set fluid- and structure-based scalar transport values required in FSI
    set_scatra_values_in_fsi();

    // solve FSI system
    if (Core::Communication::my_mpi_rank(get_comm()) == 0)
      std::cout << "\n****************************************\n               FSI "
                   "SOLVER\n****************************************\n";
    fsi_->time_step(fsi_);

    // set mesh displacement and velocity fields
    // TO DO: temporally consistent transfer of velocity and other fields,
    // for the time being, zero velocity field from structure
    set_fsi_solution();

    // solve scalar transport equation
    if (Core::Communication::my_mpi_rank(get_comm()) == 0)
      std::cout << "\n****************************************\n        SCALAR TRANSPORT "
                   "SOLVER\n****************************************\n";
    scatra_evaluate_solve_iter_update();

    // check convergence for all fields and stop iteration loop if
    // convergence is achieved overall
    stopnonliniter = convergence_check(itnum);
  }

  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FS3I::PartFS3I2Wc::set_scatra_values_in_fsi()
{
  // set scalar and thermodynamic pressure values as well as time
  // derivatives and discretization from fluid scalar in fluid
  // based on time-integration scheme
  // TO DO: check dynamic cast with "Iter" routine
  switch (fsi_->fluid_field()->tim_int_scheme())
  {
    case Inpar::FLUID::timeint_afgenalpha:
    {
      if (fsi_->fluid_field()->physical_type() == Inpar::FLUID::tempdepwater)
        fsi_->fluid_field()->set_iter_scalar_fields(
            fluid_scalar_to_fluid(scatravec_[0]->scatra_field()->phiaf()),
            fluid_scalar_to_fluid(scatravec_[0]->scatra_field()->phiam()),
            fluid_scalar_to_fluid(scatravec_[0]->scatra_field()->phidtam()),
            scatravec_[0]->scatra_field()->discretization());
      else
        fsi_->fluid_field()->set_loma_iter_scalar_fields(
            fluid_scalar_to_fluid(scatravec_[0]->scatra_field()->phiaf()),
            fluid_scalar_to_fluid(scatravec_[0]->scatra_field()->phiam()),
            fluid_scalar_to_fluid(scatravec_[0]->scatra_field()->phidtam()), nullptr,
            std::dynamic_pointer_cast<ScaTra::ScaTraTimIntLoma>(scatravec_[0]->scatra_field())
                ->therm_press_af(),
            std::dynamic_pointer_cast<ScaTra::ScaTraTimIntLoma>(scatravec_[0]->scatra_field())
                ->therm_press_am(),
            std::dynamic_pointer_cast<ScaTra::ScaTraTimIntLoma>(scatravec_[0]->scatra_field())
                ->therm_press_dt_af(),
            std::dynamic_pointer_cast<ScaTra::ScaTraTimIntLoma>(scatravec_[0]->scatra_field())
                ->therm_press_dt_am(),
            std::dynamic_pointer_cast<ScaTra::ScaTraTimIntLoma>(scatravec_[0]->scatra_field())
                ->discretization());
    }
    break;
    case Inpar::FLUID::timeint_one_step_theta:
    {
      if (fsi_->fluid_field()->physical_type() == Inpar::FLUID::tempdepwater)
        fsi_->fluid_field()->set_iter_scalar_fields(
            fluid_scalar_to_fluid(scatravec_[0]->scatra_field()->phinp()),
            fluid_scalar_to_fluid(scatravec_[0]->scatra_field()->phin()),
            fluid_scalar_to_fluid(scatravec_[0]->scatra_field()->phidtnp()),
            scatravec_[0]->scatra_field()->discretization());
      else
        fsi_->fluid_field()->set_loma_iter_scalar_fields(
            fluid_scalar_to_fluid(scatravec_[0]->scatra_field()->phinp()),
            fluid_scalar_to_fluid(scatravec_[0]->scatra_field()->phin()),
            fluid_scalar_to_fluid(scatravec_[0]->scatra_field()->phidtnp()), nullptr,
            std::dynamic_pointer_cast<ScaTra::ScaTraTimIntLoma>(scatravec_[0]->scatra_field())
                ->therm_press_np(),
            std::dynamic_pointer_cast<ScaTra::ScaTraTimIntLoma>(scatravec_[0]->scatra_field())
                ->therm_press_n(),
            std::dynamic_pointer_cast<ScaTra::ScaTraTimIntLoma>(scatravec_[0]->scatra_field())
                ->therm_press_dt_np(),
            std::dynamic_pointer_cast<ScaTra::ScaTraTimIntLoma>(scatravec_[0]->scatra_field())
                ->therm_press_dt_np(),
            std::dynamic_pointer_cast<ScaTra::ScaTraTimIntLoma>(scatravec_[0]->scatra_field())
                ->discretization());
    }
    break;
    default:
      FOUR_C_THROW("Time integration scheme not supported");
      break;
  }

  // set structure-scalar field in structure
  // (Note potential inconsistencies related to this call in case of generalized-alpha time
  // integration!)
  fsi_->structure_field()->discretization()->set_state(
      1, "scalarfield", *structure_scalar_to_structure(scatravec_[1]->scatra_field()->phinp()));
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
bool FS3I::PartFS3I2Wc::convergence_check(int itnum)
{
  // define flags for fluid and scatra convergence check
  bool fluidstopnonliniter = false;
  bool scatrastopnonliniter = false;

  // dump on screen
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
    std::cout << "\n****************************************\n  CONVERGENCE CHECK FOR ITERATION "
                 "STEP\n****************************************\n";

  // fsi convergence check
  if (fsi_->nox_status() == ::NOX::StatusTest::Converged) fluidstopnonliniter = true;

  // scatra convergence check
  scatrastopnonliniter = scatra_convergence_check(itnum);

  // warn if itemax is reached without convergence of FSI solver,
  // but proceed to next timestep
  if ((itnum == itmax_) and (fluidstopnonliniter == false))
  {
    fluidstopnonliniter = true;
    if (Core::Communication::my_mpi_rank(get_comm()) == 0)
    {
      printf("\n");
      printf(">>>>>> FSI solver not converged in itemax steps!\n");
      printf("\n");
    }
  }

  if (fluidstopnonliniter == true and scatrastopnonliniter == true)
    return true;
  else
    return false;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
bool FS3I::PartFS3I2Wc::scatra_convergence_check(int itnum)
{
  // define flags for convergence check for scatra fields
  bool scatra1stopnonliniter = false;
  bool scatra2stopnonliniter = false;

  // convergence check of scatra fields
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
  {
    std::cout << "\n****************************************\n         SCALAR TRANSPORT "
                 "CHECK\n****************************************\n";
    std::cout << "\n****************************************\n   FLUID-BASED SCALAR TRANSPORT "
                 "CHECK\n****************************************\n";
  }
  scatra1stopnonliniter =
      std::dynamic_pointer_cast<ScaTra::ScaTraTimIntLoma>(scatravec_[0]->scatra_field())
          ->convergence_check(itnum, itmax_, ittol_);

  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
    std::cout << "\n****************************************\n STRUCTURE-BASED SCALAR TRANSPORT "
                 "CHECK\n****************************************\n";
  // FOUR_C_THROW("convergence_check in scatra currently only for loma scatra!Fix this!");
  // scatra2stopnonliniter = scatravec_[1]->ScaTraField()->convergence_check(itnum,itmax_,ittol_);
  scatra2stopnonliniter =
      std::dynamic_pointer_cast<ScaTra::ScaTraTimIntLoma>(scatravec_[1]->scatra_field())
          ->convergence_check(itnum, itmax_, ittol_);

  if (scatra1stopnonliniter == true and scatra2stopnonliniter == true)
    return true;
  else
    return false;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FS3I::PartFS3I2Wc::time_update_and_output()
{
  // prepare output for FSI
  constexpr bool force_prepare = false;
  fsi_->prepare_output(force_prepare);

  // update fluid- and structure-based scalar transport
  update_scatra_fields();

  // in case of non-constant thermodynamic pressure: update
  if (consthermpress_ == "No_energy" or consthermpress_ == "No_mass")
    std::dynamic_pointer_cast<ScaTra::ScaTraTimIntLoma>(scatravec_[0]->scatra_field())
        ->update_therm_pressure();

  // update structure, fluid and ALE
  fsi_->update();

  // set scalar and thermodynamic pressure at n+1 and SCATRA trueresidual
  // for statistical evaluation and evaluation of Neumann boundary
  // conditions at the beginning of the subsequent time step
  fsi_->fluid_field()->set_scalar_fields(
      fluid_scalar_to_fluid(scatravec_[0]->scatra_field()->phinp()),
      std::dynamic_pointer_cast<ScaTra::ScaTraTimIntLoma>(scatravec_[0]->scatra_field())
          ->therm_press_np(),
      fluid_scalar_to_fluid(scatravec_[0]->scatra_field()->true_residual()),
      scatravec_[0]->scatra_field()->discretization());

  // Note: The order is important here! Herein, control file entries are
  // written, defining the order in which the filters handle the
  // discretizations, which in turn defines the dof number ordering of the
  // discretizations.
  fsi_->output();

  // output of fluid- and structure-based scalar transport
  scatra_output();

  return;
}

FOUR_C_NAMESPACE_CLOSE
