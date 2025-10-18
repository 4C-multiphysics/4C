// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_pasi_partitioned_twowaycoup.hpp"

#include "4C_adapter_str_pasiwrapper.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_particle_algorithm.hpp"
#include "4C_particle_wall_datastate.hpp"
#include "4C_particle_wall_interface.hpp"
#include "4C_structure_aux.hpp"

#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
PaSI::PasiPartTwoWayCoup::PasiPartTwoWayCoup(MPI_Comm comm, const Teuchos::ParameterList& params)
    : PartitionedAlgo(comm, params),
      itmax_(params.get<int>("ITEMAX")),
      convtolrelativedisp_(params.get<double>("CONVTOLRELATIVEDISP")),
      convtolscaleddisp_(params.get<double>("CONVTOLSCALEDDISP")),
      convtolrelativeforce_(params.get<double>("CONVTOLRELATIVEFORCE")),
      convtolscaledforce_(params.get<double>("CONVTOLSCALEDFORCE")),
      ignoreconvcheck_(params.get<bool>("IGNORE_CONV_CHECK")),
      writerestartevery_(params.get<int>("RESTARTEVERY"))
{
  // empty constructor
}

void PaSI::PasiPartTwoWayCoup::init()
{
  // call base class init
  PaSI::PartitionedAlgo::init();

  // construct interface force
  intfforcenp_ = Core::LinAlg::create_vector(*interface_->pasi_cond_map(), true);

  // construct interface increment states
  intfdispincnp_ = Core::LinAlg::create_vector(*interface_->pasi_cond_map(), true);
  intfforceincnp_ = Core::LinAlg::create_vector(*interface_->pasi_cond_map(), true);

  // safety check
  if (convtolrelativedisp_ < 0.0 and convtolscaleddisp_ < 0.0 and convtolrelativeforce_ < 0.0 and
      convtolscaledforce_ < 0.0)
    FOUR_C_THROW("no convergence tolerance for partitioned iterations set!");
}

void PaSI::PasiPartTwoWayCoup::setup()
{
  // call base class setup
  PaSI::PartitionedAlgo::setup();

  // safety check
  {
    // get interface to particle wall handler
    std::shared_ptr<PARTICLEWALL::WallHandlerInterface> particlewallinterface =
        particlealgorithm_->get_particle_wall_handler_interface();

    // get wall data state container
    std::shared_ptr<PARTICLEWALL::WallDataState> walldatastate =
        particlewallinterface->get_wall_data_state();

    if (walldatastate->get_disp_row() == nullptr or walldatastate->get_disp_col() == nullptr)
      FOUR_C_THROW("wall displacements not initialized!");
    if (walldatastate->get_vel_col() == nullptr) FOUR_C_THROW("wall velocities not initialized!");
    if (walldatastate->get_acc_col() == nullptr)
      FOUR_C_THROW("wall accelerations not initialized!");
    if (walldatastate->get_force_col() == nullptr) FOUR_C_THROW("wall forces not initialized!");
  }
}

void PaSI::PasiPartTwoWayCoup::read_restart(int restartstep)
{
  // call base class read restart
  PaSI::PartitionedAlgo::read_restart(restartstep);

  Core::IO::DiscretizationReader reader(structurefield_->discretization(),
      Global::Problem::instance()->input_control_file(), restartstep);
  if (restartstep != reader.read_int("step"))
    FOUR_C_THROW("Time step on file not equal to given step");

  // get interface force from restart
  reader.read_vector(intfforcenp_, "intfforcenp");
}

void PaSI::PasiPartTwoWayCoup::timeloop()
{
  // safety checks
  check_is_init();
  check_is_setup();

  while (not_finished())
  {
    // prepare time step
    prepare_time_step();

    // pre evaluate time step
    pre_evaluate_time_step();

    // iteration loop between coupled fields
    outerloop();

    // post evaluate time step
    post_evaluate_time_step();

    // output of fields
    output();
  }
}

void PaSI::PasiPartTwoWayCoup::outerloop()
{
  int itnum = 0;
  bool stopnonliniter = false;

  if ((Core::Communication::my_mpi_rank(get_comm()) == 0) and print_screen_every() and
      (step() % print_screen_every() == 0))
  {
    // clang-format off
    printf("+------------------------------------------------------------------------------+\n");
    printf("|  ITERATION LOOP BETWEEN COUPLED FIELDS                                       |\n");
    printf("+------------------------------------------------------------------------------+\n");
    // clang-format on
  }

  // save particle states
  save_particle_states();

  while (stopnonliniter == false)
  {
    // increment number of iteration
    ++itnum;

    // reset increment states
    reset_increment_states(*intfdispnp_, *intfforcenp_);

    // reset particle states
    reset_particle_states();

    // clear interface forces
    clear_interface_forces();

    // particle time step
    particle_step();

    // get interface forces
    get_interface_forces();

    // set interface forces
    set_interface_forces(intfforcenp_);

    // structural time step
    struct_step();

    // extract interface states
    extract_interface_states();

    // build increment states
    build_increment_states();

    // convergence check for structure and particles fields
    stopnonliniter = convergence_check(itnum);

    // set interface states
    set_interface_states(intfdispnp_, intfvelnp_, intfaccnp_);
  }
}

void PaSI::PasiPartTwoWayCoup::output()
{
  // output of structure field
  struct_output();

  // write interface force in restart
  if (writerestartevery_ and step() % writerestartevery_ == 0)
    structurefield_->discretization()->writer()->write_vector("intfforcenp", intfforcenp_);

  // output of particle field
  particle_output();
}

void PaSI::PasiPartTwoWayCoup::reset_increment_states(
    const Core::LinAlg::Vector<double>& intfdispnp, const Core::LinAlg::Vector<double>& intfforcenp)
{
  intfdispincnp_->update(1.0, intfdispnp, 0.0);
  intfforceincnp_->update(1.0, intfforcenp, 0.0);
}

void PaSI::PasiPartTwoWayCoup::build_increment_states()
{
  intfdispincnp_->update(1.0, *intfdispnp_, -1.0);
  intfforceincnp_->update(1.0, *intfforcenp_, -1.0);
}

void PaSI::PasiPartTwoWayCoup::set_interface_forces(
    std::shared_ptr<const Core::LinAlg::Vector<double>> intfforcenp)
{
  TEUCHOS_FUNC_TIME_MONITOR("PaSI::PASI_PartTwoWayCoup::set_interface_forces");

  // apply interface force on structure discretization
  structurefield_->apply_interface_force(intfforcenp);

  // print norm of interface force to the screen
  if (print_screen_every() and (step() % print_screen_every() == 0))
  {
    double normintfforce(0.0);
    intfforcenp->norm_2(&normintfforce);

    if (Core::Communication::my_mpi_rank(get_comm()) == 0)
      printf("--> Norm of interface force: %10.5E\n", normintfforce);
  }
}

void PaSI::PasiPartTwoWayCoup::reset_particle_states()
{
  TEUCHOS_FUNC_TIME_MONITOR("PaSI::PASI_PartTwoWayCoup::reset_particle_states");

  // get interface to particle engine
  std::shared_ptr<PARTICLEENGINE::ParticleEngineInterface> particleengineinterface =
      particlealgorithm_->get_particle_engine_interface();

  // get particle container bundle
  PARTICLEENGINE::ParticleContainerBundleShrdPtr particlecontainerbundle =
      particleengineinterface->get_particle_container_bundle();

  // iterate over particle types
  for (const auto& type : particlecontainerbundle->get_particle_types())
  {
    // get container of owned particles of current particle type
    PARTICLEENGINE::ParticleContainer* container =
        particlecontainerbundle->get_specific_container(type, PARTICLEENGINE::Owned);

    // reset position, velocity and acceleration states of all particles
    container->update_state(0.0, PARTICLEENGINE::Position, 1.0, PARTICLEENGINE::LastIterPosition);
    container->update_state(0.0, PARTICLEENGINE::Velocity, 1.0, PARTICLEENGINE::LastIterVelocity);
    container->update_state(
        0.0, PARTICLEENGINE::Acceleration, 1.0, PARTICLEENGINE::LastIterAcceleration);

    // reset angular velocity state of all particles
    if (container->have_stored_state(PARTICLEENGINE::AngularVelocity))
      container->update_state(
          0.0, PARTICLEENGINE::AngularVelocity, 1.0, PARTICLEENGINE::LastIterAngularVelocity);

    // reset angular acceleration state of all particles
    if (container->have_stored_state(PARTICLEENGINE::AngularAcceleration))
      container->update_state(0.0, PARTICLEENGINE::AngularAcceleration, 1.0,
          PARTICLEENGINE::LastIterAngularAcceleration);

    // reset modified acceleration state of all particles
    if (container->have_stored_state(PARTICLEENGINE::ModifiedAcceleration))
      container->update_state(0.0, PARTICLEENGINE::ModifiedAcceleration, 1.0,
          PARTICLEENGINE::LastIterModifiedAcceleration);

    // reset density state of all particles
    if (container->have_stored_state(PARTICLEENGINE::DensityDot))
      container->update_state(0.0, PARTICLEENGINE::Density, 1.0, PARTICLEENGINE::LastIterDensity);

    // reset temperature state of all particles
    if (container->have_stored_state(PARTICLEENGINE::TemperatureDot))
      container->update_state(
          0.0, PARTICLEENGINE::Temperature, 1.0, PARTICLEENGINE::LastIterTemperature);
  }
}

void PaSI::PasiPartTwoWayCoup::clear_interface_forces()
{
  TEUCHOS_FUNC_TIME_MONITOR("PaSI::PASI_PartTwoWayCoup::clear_interface_forces");

  // get interface to particle wall handler
  std::shared_ptr<PARTICLEWALL::WallHandlerInterface> particlewallinterface =
      particlealgorithm_->get_particle_wall_handler_interface();

  // get wall data state container
  std::shared_ptr<PARTICLEWALL::WallDataState> walldatastate =
      particlewallinterface->get_wall_data_state();

#ifdef FOUR_C_ENABLE_ASSERTIONS
  if (walldatastate->get_force_col() == nullptr) FOUR_C_THROW("wall forces not initialized!");
#endif

  // clear interface forces
  walldatastate->get_force_col()->put_scalar(0.0);
}

void PaSI::PasiPartTwoWayCoup::get_interface_forces()
{
  TEUCHOS_FUNC_TIME_MONITOR("PaSI::PASI_PartTwoWayCoup::get_interface_forces");

  // get interface to particle wall handler
  std::shared_ptr<PARTICLEWALL::WallHandlerInterface> particlewallinterface =
      particlealgorithm_->get_particle_wall_handler_interface();

  // get wall data state container
  std::shared_ptr<PARTICLEWALL::WallDataState> walldatastate =
      particlewallinterface->get_wall_data_state();

#ifdef FOUR_C_ENABLE_ASSERTIONS
  if (walldatastate->get_force_col() == nullptr) FOUR_C_THROW("wall forces not initialized!");
#endif

  // clear interface forces
  intfforcenp_->put_scalar(0.0);

  // assemble interface forces
  Core::LinAlg::Export exporter(walldatastate->get_force_col()->get_map(), intfforcenp_->get_map());
  intfforcenp_->export_to(*walldatastate->get_force_col(), exporter, Add);
}

bool PaSI::PasiPartTwoWayCoup::convergence_check(int itnum)
{
  bool stopnonliniter = false;

  // variables to save different L2-Norms
  double intfdispincnorm_L2(0.0);
  double intfdispnorm_L2(0.0);
  double intfforceincnorm_L2(0.0);
  double intfforcenorm_L2(0.0);

  // build L2-norm of interface displacement increment and interface displacement
  intfdispincnp_->norm_2(&intfdispincnorm_L2);
  intfdispnp_->norm_2(&intfdispnorm_L2);

  // build L2-norm of interface force increment and interface force
  intfforceincnp_->norm_2(&intfforceincnorm_L2);
  intfforcenp_->norm_2(&intfforcenorm_L2);

  // care for the case that there is (almost) zero scalar
  if (intfdispnorm_L2 < 1e-6) intfdispnorm_L2 = 1.0;
  if (intfforcenorm_L2 < 1e-6) intfforcenorm_L2 = 1.0;

  // scaled and relative interface displacement increment
  double scaled_disp_inc = intfdispincnorm_L2 / (dt() * sqrt(intfdispincnp_->global_length()));
  double relative_disp_inc = intfdispincnorm_L2 / intfdispnorm_L2;

  // scaled and relative interface force increment
  double scaled_force_inc = intfforceincnorm_L2 / (dt() * sqrt(intfforceincnp_->global_length()));
  double relative_force_inc = intfforceincnorm_L2 / intfforcenorm_L2;

  // print the incremental based convergence check to the screen
  if ((Core::Communication::my_mpi_rank(get_comm()) == 0) and print_screen_every() and
      (step() % print_screen_every() == 0))
  {
    // clang-format off
    printf("+----------+-----------------+--------------+------------------+---------------+\n");
    printf("| step/max | scaled-disp-inc | rel-disp-inc | scaled-force-inc | rel-force-inc |\n");
    printf("|  %3d/%3d |      %10.3E |   %10.3E |       %10.3E |    %10.3E |\n", itnum, itmax_, scaled_disp_inc, relative_disp_inc, scaled_force_inc, relative_force_inc);
    printf("+----------+-----------------+--------------+------------------+---------------+\n");
    // clang-format on
  }

  bool isconverged = true;

  // check convergence of scaled interface displacement increment
  if (convtolscaleddisp_ > 0.0) isconverged &= scaled_disp_inc <= convtolscaleddisp_;

  // check convergence of relative interface displacement increment
  if (convtolrelativedisp_ > 0.0) isconverged &= relative_disp_inc <= convtolrelativedisp_;

  // check convergence of scaled interface force increment
  if (convtolscaledforce_ > 0.0) isconverged &= scaled_force_inc <= convtolscaledforce_;

  // check convergence of relative interface force increment
  if (convtolrelativeforce_ > 0.0) isconverged &= relative_force_inc <= convtolrelativeforce_;

  // converged
  if (isconverged)
  {
    stopnonliniter = true;

    if ((Core::Communication::my_mpi_rank(get_comm()) == 0) and print_screen_every() and
        (step() % print_screen_every() == 0))
    {
      // clang-format off
      printf("|  Outer iteration loop converged after iteration %3d/%3d !                    |\n", itnum, itmax_);
      printf("+------------------------------------------------------------------------------+\n");
      // clang-format on
    }
  }

  // stop if maximum iteration number is reached without convergence
  if ((itnum == itmax_) and (not isconverged))
  {
    stopnonliniter = true;

    // ignore convergence check and proceed simulation
    if (ignoreconvcheck_)
    {
      if ((Core::Communication::my_mpi_rank(get_comm()) == 0) and print_screen_every() and
          (step() % print_screen_every() == 0))
      {
        // clang-format off
        printf("|  ATTENTION: Outer iteration loop not converged in itemax = %3d steps!        |\n", itmax_);
        printf("+------------------------------------------------------------------------------+\n");
        // clang-format on
      }
    }
    // abort the simulation
    else
    {
      if ((Core::Communication::my_mpi_rank(get_comm()) == 0) and print_screen_every() and
          (step() % print_screen_every() == 0))
      {
        // clang-format off
        printf("|  STOP: Outer iteration loop not converged in itemax = %3d steps              |\n", itmax_);
        printf("+------------------------------------------------------------------------------+\n");
        // clang-format on
      }
      FOUR_C_THROW("The partitioned PASI solver did not converge in ITEMAX steps!");
    }
  }

  return stopnonliniter;
}

void PaSI::PasiPartTwoWayCoup::save_particle_states()
{
  TEUCHOS_FUNC_TIME_MONITOR("PaSI::PASI_PartTwoWayCoup::save_particle_states");

  // get interface to particle engine
  std::shared_ptr<PARTICLEENGINE::ParticleEngineInterface> particleengineinterface =
      particlealgorithm_->get_particle_engine_interface();

  // get particle container bundle
  PARTICLEENGINE::ParticleContainerBundleShrdPtr particlecontainerbundle =
      particleengineinterface->get_particle_container_bundle();

  // iterate over particle types
  for (const auto& type : particlecontainerbundle->get_particle_types())
  {
    // get container of owned particles of current particle type
    PARTICLEENGINE::ParticleContainer* container =
        particlecontainerbundle->get_specific_container(type, PARTICLEENGINE::Owned);

    // save position, velocity and acceleration states of all particles
    container->update_state(0.0, PARTICLEENGINE::LastIterPosition, 1.0, PARTICLEENGINE::Position);
    container->update_state(0.0, PARTICLEENGINE::LastIterVelocity, 1.0, PARTICLEENGINE::Velocity);
    container->update_state(
        0.0, PARTICLEENGINE::LastIterAcceleration, 1.0, PARTICLEENGINE::Acceleration);

    // save angular velocity state of all particles
    if (container->have_stored_state(PARTICLEENGINE::AngularVelocity))
      container->update_state(
          0.0, PARTICLEENGINE::LastIterAngularVelocity, 1.0, PARTICLEENGINE::AngularVelocity);

    // save angular acceleration state of all particles
    if (container->have_stored_state(PARTICLEENGINE::AngularAcceleration))
      container->update_state(0.0, PARTICLEENGINE::LastIterAngularAcceleration, 1.0,
          PARTICLEENGINE::AngularAcceleration);

    // save modified acceleration state of all particles
    if (container->have_stored_state(PARTICLEENGINE::ModifiedAcceleration))
      container->update_state(0.0, PARTICLEENGINE::LastIterModifiedAcceleration, 1.0,
          PARTICLEENGINE::ModifiedAcceleration);

    // save density state of all particles
    if (container->have_stored_state(PARTICLEENGINE::DensityDot))
      container->update_state(0.0, PARTICLEENGINE::LastIterDensity, 1.0, PARTICLEENGINE::Density);

    // save temperature state of all particles
    if (container->have_stored_state(PARTICLEENGINE::TemperatureDot))
      container->update_state(
          0.0, PARTICLEENGINE::LastIterTemperature, 1.0, PARTICLEENGINE::Temperature);
  }
}

PaSI::PasiPartTwoWayCoupDispRelax::PasiPartTwoWayCoupDispRelax(
    MPI_Comm comm, const Teuchos::ParameterList& params)
    : PasiPartTwoWayCoup(comm, params), omega_(params.get<double>("STARTOMEGA"))
{
  // empty constructor
}

void PaSI::PasiPartTwoWayCoupDispRelax::init()
{
  // call base class init
  PaSI::PasiPartTwoWayCoup::init();

  // construct relaxed interface states
  relaxintfdispnp_ = Core::LinAlg::create_vector(*interface_->pasi_cond_map(), true);
  relaxintfvelnp_ = Core::LinAlg::create_vector(*interface_->pasi_cond_map(), true);
  relaxintfaccnp_ = Core::LinAlg::create_vector(*interface_->pasi_cond_map(), true);
}

void PaSI::PasiPartTwoWayCoupDispRelax::outerloop()
{
  int itnum = 0;
  bool stopnonliniter = false;

  if ((Core::Communication::my_mpi_rank(get_comm()) == 0) and print_screen_every() and
      (step() % print_screen_every() == 0))
  {
    // clang-format off
    printf("+------------------------------------------------------------------------------+\n");
    printf("|  ITERATION LOOP BETWEEN COUPLED FIELDS WITH RELAXED DISPLACEMENTS            |\n");
    printf("+------------------------------------------------------------------------------+\n");
    // clang-format on
  }

  // init relaxation of interface states
  init_relaxation_interface_states();

  // set interface states
  set_interface_states(relaxintfdispnp_, relaxintfvelnp_, relaxintfaccnp_);

  // save particle states
  save_particle_states();

  while (stopnonliniter == false)
  {
    // increment number of iteration
    ++itnum;

    // reset increment states
    reset_increment_states(*relaxintfdispnp_, *intfforcenp_);

    // reset particle states
    reset_particle_states();

    // clear interface forces
    clear_interface_forces();

    // particle time step
    particle_step();

    // get interface forces
    get_interface_forces();

    // set interface forces
    set_interface_forces(intfforcenp_);

    // structural time step
    struct_step();

    // extract interface states
    extract_interface_states();

    // build increment states
    build_increment_states();

    // convergence check for structure and particles fields
    stopnonliniter = convergence_check(itnum);

    // calculate relaxation parameter
    calc_omega(omega_, itnum);

    // perform relaxation of interface states
    perform_relaxation_interface_states();

    // set interface states
    set_interface_states(relaxintfdispnp_, relaxintfvelnp_, relaxintfaccnp_);
  }
}

void PaSI::PasiPartTwoWayCoupDispRelax::calc_omega(double& omega, const int itnum)
{
  // output constant relaxation parameter
  if ((Core::Communication::my_mpi_rank(get_comm()) == 0) and print_screen_every() and
      (step() % print_screen_every() == 0))
    std::cout << "Fixed relaxation parameter: " << omega << std::endl;
}

void PaSI::PasiPartTwoWayCoupDispRelax::init_relaxation_interface_states()
{
  relaxintfdispnp_->update(1.0, *intfdispnp_, 0.0);
  relaxintfvelnp_->update(1.0, *intfvelnp_, 0.0);
  relaxintfaccnp_->update(1.0, *intfaccnp_, 0.0);
}

void PaSI::PasiPartTwoWayCoupDispRelax::perform_relaxation_interface_states()
{
  relaxintfdispnp_->update(omega_, *intfdispincnp_, 1.0);

  relaxintfvelnp_->update(1.0, *intfdispnp_, 0.0);
  relaxintfvelnp_->update(1.0 / dt(), *relaxintfdispnp_, -1.0 / dt());

  relaxintfaccnp_->update(1.0, *intfvelnp_, 0.0);
  relaxintfaccnp_->update(1.0 / dt(), *relaxintfvelnp_, -1.0 / dt());
}

PaSI::PasiPartTwoWayCoupDispRelaxAitken::PasiPartTwoWayCoupDispRelaxAitken(
    MPI_Comm comm, const Teuchos::ParameterList& params)
    : PasiPartTwoWayCoupDispRelax(comm, params),
      maxomega_(params.get<double>("MAXOMEGA")),
      minomega_(params.get<double>("MINOMEGA"))
{
  // empty constructor
}

void PaSI::PasiPartTwoWayCoupDispRelaxAitken::init()
{
  // call base class init
  PaSI::PasiPartTwoWayCoupDispRelax::init();

  // construct old interface increment state
  intfdispincnpold_ = Core::LinAlg::create_vector(*interface_->pasi_cond_map(), true);
}

void PaSI::PasiPartTwoWayCoupDispRelaxAitken::read_restart(int restartstep)
{
  // call base class read restart
  PaSI::PasiPartTwoWayCoupDispRelax::read_restart(restartstep);

  Core::IO::DiscretizationReader reader(structurefield_->discretization(),
      Global::Problem::instance()->input_control_file(), restartstep);
  if (restartstep != reader.read_int("step"))
    FOUR_C_THROW("Time step on file not equal to given step");

  // get relaxation parameter from restart
  omega_ = reader.read_double("omega");
}

void PaSI::PasiPartTwoWayCoupDispRelaxAitken::output()
{
  // output of structure field
  struct_output();

  // write interface force and relaxation parameter in restart
  if (writerestartevery_ and step() % writerestartevery_ == 0)
  {
    structurefield_->discretization()->writer()->write_vector("intfforcenp", intfforcenp_);
    structurefield_->discretization()->writer()->write_double("omega", omega_);
  }

  // output of particle field
  particle_output();
}

void PaSI::PasiPartTwoWayCoupDispRelaxAitken::calc_omega(double& omega, const int itnum)
{
  std::shared_ptr<Core::LinAlg::Vector<double>> intfdispincnpdiff =
      Core::LinAlg::create_vector(*interface_->pasi_cond_map(), true);
  intfdispincnpdiff->update(1.0, *intfdispincnp_, (-1.0), *intfdispincnpold_, 0.0);

  double dispincnpdiffnorm(0.0);
  intfdispincnpdiff->norm_2(&dispincnpdiffnorm);

  if (dispincnpdiffnorm <= 1e-06)
  {
    if ((Core::Communication::my_mpi_rank(get_comm()) == 0) and print_screen_every() and
        (step() % print_screen_every() == 0))
      std::cout << "Warning: The norm of displacement increment is to small to use it for Aitken "
                   "relaxation. Reuse previous Aitken relaxation parameter instead!"
                << std::endl;
  }

  // in first iteration reuse Aitken relaxation parameter from previous step
  if (itnum != 1 and dispincnpdiffnorm > 1e-06)
  {
    double dispincsdot(0.0);
    intfdispincnpdiff->dot(*intfdispincnp_, &dispincsdot);

    // update Aitken relaxation parameter
    omega = omega * (1.0 - (dispincsdot) / (dispincnpdiffnorm * dispincnpdiffnorm));

    // allowed range for Aitken relaxation parameter
    if (omega < minomega_)
    {
      if ((Core::Communication::my_mpi_rank(get_comm()) == 0) and print_screen_every() and
          (step() % print_screen_every() == 0))
        std::cout << "Warning: The calculation of the relaxation parameter via Aitken did lead to "
                     "a value smaller than MINOMEGA!"
                  << std::endl;
      omega = minomega_;
    }
    if (omega > maxomega_)
    {
      if ((Core::Communication::my_mpi_rank(get_comm()) == 0) and print_screen_every() and
          (step() % print_screen_every() == 0))
        std::cout << "Warning: The calculation of the relaxation parameter via Aitken did lead to "
                     "a value bigger than MAXOMEGA!"
                  << std::endl;
      omega = maxomega_;
    }
  }

  // output Aitken relaxation parameter
  if ((Core::Communication::my_mpi_rank(get_comm()) == 0) and print_screen_every() and
      (step() % print_screen_every() == 0))
    std::cout << "Aitken relaxation parameter: " << omega << std::endl;

  // store current interface displacement increment for next iteration
  intfdispincnpold_->update(1.0, *intfdispincnp_, 0.0);
}

FOUR_C_NAMESPACE_CLOSE
