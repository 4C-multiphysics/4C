// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fs3i_partitioned_1wc.hpp"

#include "4C_adapter_fld_fluid_fsi.hpp"
#include "4C_adapter_str_fsiwrapper.hpp"
#include "4C_comm_mpi_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fsi_monolithic.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_scatra.hpp"
#include "4C_scatra_algorithm.hpp"
#include "4C_scatra_timint_implicit.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
FS3I::PartFS3I1Wc::PartFS3I1Wc(MPI_Comm comm) : PartFS3I(comm) {}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FS3I::PartFS3I1Wc::init()
{
  FS3I::PartFS3I::init();
  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FS3I::PartFS3I1Wc::setup()
{
  FS3I::PartFS3I::setup();
  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FS3I::PartFS3I1Wc::timeloop()
{
  check_is_init();
  check_is_setup();

  // prepare time loop
  fsi_->prepare_timeloop();
  set_fsi_solution();

  // calculate initial time derivative, when restart was done from a part. FSI simulation
  if (static_cast<bool>(Global::Problem::instance()->restart()) and
      Global::Problem::instance()->f_s3_i_dynamic_params().get<bool>("RESTART_FROM_PART_FSI"))
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
    set_struct_scatra_solution();
    do_fsi_step();
    set_fsi_solution();
    do_scatra_step();
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FS3I::PartFS3I1Wc::do_fsi_step()
{
  fsi_->prepare_time_step();
  fsi_->time_step(fsi_);
  constexpr bool force_prepare = false;
  fsi_->prepare_output(force_prepare);
  fsi_->update();
  fsi_->output();
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FS3I::PartFS3I1Wc::do_scatra_step()
{
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
  {
    std::cout << "\n***********************\n GAS TRANSPORT SOLVER \n***********************\n"
              << std::endl;
    std::cout << "+- step/max -+- abs-res-tol [norm] -+-- scal-res --+- rel-inc-tol [norm] -+-- "
                 "scal-inc --+"
              << std::endl;
  }

  // first scatra field is associated with fluid, second scatra field is
  // associated with structure

  bool stopnonliniter = false;
  int itnum = 0;

  prepare_time_step();

  while (stopnonliniter == false)
  {
    scatra_evaluate_solve_iter_update();
    itnum++;
    if (scatra_convergence_check(itnum)) break;
  }

  update_scatra_fields();
  scatra_output();
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FS3I::PartFS3I1Wc::prepare_time_step()
{
  check_is_init();
  check_is_setup();

  // set mesh displacement field for present time step
  set_mesh_disp();

  // set velocity fields from fluid and structure solution
  // for present time step
  set_velocity_fields();

  // prepare time step for both fluid- and structure-based scatra field
  for (unsigned i = 0; i < scatravec_.size(); ++i)
  {
    std::shared_ptr<Adapter::ScaTraBaseAlgorithm> scatra = scatravec_[i];
    scatra->scatra_field()->prepare_time_step();
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
bool FS3I::PartFS3I1Wc::scatra_convergence_check(const int itnum)
{
  const Teuchos::ParameterList& fs3idyn = Global::Problem::instance()->f_s3_i_dynamic_params();
  auto scatra_solvtype =
      Teuchos::getIntegralValue<Inpar::ScaTra::SolverType>(fs3idyn, "SCATRA_SOLVERTYPE");

  double conresnorm(0.0);
  scatrarhs_->Norm2(&conresnorm);
  double incconnorm(0.0);
  scatraincrement_->Norm2(&incconnorm);

  switch (scatra_solvtype)
  {
    case Inpar::ScaTra::solvertype_linear_incremental:
    {
      // print the screen info
      if (Core::Communication::my_mpi_rank(get_comm()) == 0)
      {
        printf("\n+-------------------+-------------------+\n");
        printf("| norm of residual  | norm of increment |\n");
        printf("+-------------------+-------------------+\n");
        printf("|    %10.3E     |    %10.3E     |\n", conresnorm, incconnorm);
        printf("+-------------------+-------------------+\n\n");
      }
      return true;
    }
    break;
    case Inpar::ScaTra::solvertype_nonlinear:
    {
      // some input parameters for the scatra fields
      const Teuchos::ParameterList& scatradyn =
          Global::Problem::instance()->scalar_transport_dynamic_params();
      const int itemax = scatradyn.sublist("NONLINEAR").get<int>("ITEMAX");
      const double ittol = scatradyn.sublist("NONLINEAR").get<double>("CONVTOL");
      const double abstolres = scatradyn.sublist("NONLINEAR").get<double>("ABSTOLRES");

      double connorm(0.0);
      // set up vector of absolute concentrations
      Core::LinAlg::Vector<double> con(scatraincrement_->Map());
      std::shared_ptr<const Core::LinAlg::Vector<double>> scatra1 =
          scatravec_[0]->scatra_field()->phinp();
      std::shared_ptr<const Core::LinAlg::Vector<double>> scatra2 =
          scatravec_[1]->scatra_field()->phinp();
      setup_coupled_scatra_vector(con, *scatra1, *scatra2);
      con.Norm2(&connorm);

      // care for the case that nothing really happens in the concentration field
      if (connorm < 1e-5) connorm = 1.0;

      // print the screen info
      if (Core::Communication::my_mpi_rank(get_comm()) == 0)
      {
        printf("|  %3d/%3d   |   %10.3E [L_2 ]  | %10.3E   |   %10.3E [L_2 ]  | %10.3E   |\n",
            itnum, itemax, abstolres, conresnorm, ittol, incconnorm / connorm);
      }

      // this is the convergence check
      // We always require at least one solve. We test the L_2-norm of the
      // current residual. Norm of residual is just printed for information
      if (conresnorm <= abstolres and incconnorm / connorm <= ittol)
      {
        if (Core::Communication::my_mpi_rank(get_comm()) == 0)
        {
          // print 'finish line'
          printf(
              "+------------+----------------------+--------------+----------------------+---------"
              "-----+\n\n");
        }
        return true;
      }
      // warn if itemax is reached without convergence, but proceed to
      // next timestep...
      else if (itnum == itemax)
      {
        if (Core::Communication::my_mpi_rank(get_comm()) == 0)
        {
          printf("+---------------------------------------------------------------+\n");
          printf("|            >>>>>> not converged in itemax steps!              |\n");
          printf("+---------------------------------------------------------------+\n");
        }
        // yes, we stop the iteration
        return true;
      }
      else
        return false;
    }
    break;
    default:
      FOUR_C_THROW("Illegal ScaTra solvertype in FS3I");
      break;
  }
  return false;
}

FOUR_C_NAMESPACE_CLOSE
