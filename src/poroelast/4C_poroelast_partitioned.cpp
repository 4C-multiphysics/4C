// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_poroelast_partitioned.hpp"

#include "4C_adapter_fld_poro.hpp"
#include "4C_adapter_str_fpsiwrapper.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_structure_aux.hpp"

FOUR_C_NAMESPACE_OPEN

PoroElast::Partitioned::Partitioned(const Epetra_Comm& comm,
    const Teuchos::ParameterList& timeparams,
    Teuchos::RCP<Core::LinAlg::MapExtractor> porosity_splitter)
    : PoroBase(comm, timeparams, porosity_splitter),
      fluidincnp_(Teuchos::make_rcp<Core::LinAlg::Vector<double>>(*(fluid_field()->velnp()))),
      structincnp_(Teuchos::make_rcp<Core::LinAlg::Vector<double>>(*(structure_field()->dispnp())))
{
  const Teuchos::ParameterList& porodyn = Global::Problem::instance()->poroelast_dynamic_params();
  // Get the parameters for the convergence_check
  itmax_ = porodyn.get<int>("ITEMAX");     // default: =10
  ittol_ = porodyn.get<double>("INCTOL");  // default: =1e-6

  fluidveln_ = Core::LinAlg::create_vector(*(fluid_field()->dof_row_map()), true);
  fluidveln_->PutScalar(0.0);
}

void PoroElast::Partitioned::do_time_step()
{
  prepare_time_step();

  solve();

  update_and_output();
}

void PoroElast::Partitioned::setup_system() {}  // SetupSystem()

void PoroElast::Partitioned::update_and_output()
{
  constexpr bool force_prepare = false;
  prepare_output(force_prepare);

  update();

  output();
}

void PoroElast::Partitioned::solve()
{
  int itnum = 0;
  bool stopnonliniter = false;

  if (get_comm().MyPID() == 0)
  {
    std::cout << "\n****************************************\n          OUTER ITERATION "
                 "LOOP\n****************************************\n";
  }

  if (step() == 1)
  {
    fluidveln_->Update(1.0, *(fluid_field()->veln()), 0.0);
  }

  while (!stopnonliniter)
  {
    itnum++;

    // store increment from first solution for convergence check (like in
    // elch_algorithm: use current values)
    fluidincnp_->Update(1.0, *fluid_field()->velnp(), 0.0);
    structincnp_->Update(1.0, *structure_field()->dispnp(), 0.0);

    // get current fluid velocities due to solve fluid step, like predictor in FSI
    // 1. iteration: get velocities of old time step (T_n)
    if (itnum == 1)
    {
      fluidveln_->Update(1.0, *(fluid_field()->veln()), 0.0);
    }
    else  // itnum > 1
    {
      // save velocity solution of old iteration step T_{n+1}^i
      fluidveln_->Update(1.0, *(fluid_field()->velnp()), 0.0);
    }

    // set fluid- and structure-based scalar transport values required in FSI
    set_fluid_solution();

    if (itnum != 1) structure_field()->prepare_partition_step();
    // solve structural system
    do_struct_step();

    // set mesh displacement and velocity fields
    set_struct_solution();

    // solve scalar transport equation
    do_fluid_step();

    // check convergence for all fields and stop iteration loop if
    // convergence is achieved overall
    stopnonliniter = convergence_check(itnum);
  }
}

void PoroElast::Partitioned::do_struct_step()
{
  if (get_comm().MyPID() == 0)
  {
    std::cout << "\n***********************\n STRUCTURE SOLVER \n***********************\n";
  }

  // Newton-Raphson iteration
  structure_field()->solve();
}

void PoroElast::Partitioned::do_fluid_step()
{
  if (get_comm().MyPID() == 0)
  {
    std::cout << "\n***********************\n FLUID SOLVER \n***********************\n";
  }

  // fluid_field()->PrepareSolve();
  // Newton-Raphson iteration
  fluid_field()->solve();
}

void PoroElast::Partitioned::prepare_time_step()
{
  increment_time_and_step();
  print_header();

  structure_field()->prepare_time_step();
  set_struct_solution();
  fluid_field()->prepare_time_step();
  set_fluid_solution();
}

bool PoroElast::Partitioned::convergence_check(int itnum)
{
  // convergence check based on the increment
  bool stopnonliniter = false;

  // variables to save different L2 - Norms
  // define L2-norm of increments and solution
  double fluidincnorm_L2(0.0);
  double fluidnorm_L2(0.0);
  double dispincnorm_L2(0.0);
  double structnorm_L2(0.0);

  // build the current increment
  fluidincnp_->Update(1.0, *(fluid_field()->velnp()), -1.0);
  structincnp_->Update(1.0, *(structure_field()->dispnp()), -1.0);

  // build the L2-norm of the increment and the solution
  fluidincnp_->Norm2(&fluidincnorm_L2);
  fluid_field()->velnp()->Norm2(&fluidnorm_L2);
  structincnp_->Norm2(&dispincnorm_L2);
  structure_field()->dispnp()->Norm2(&structnorm_L2);

  // care for the case that there is (almost) zero solution
  if (fluidnorm_L2 < 1e-6) fluidnorm_L2 = 1.0;
  if (structnorm_L2 < 1e-6) structnorm_L2 = 1.0;

  // print the incremental based convergence check to the screen
  if (get_comm().MyPID() == 0)
  {
    std::cout << "\n";
    std::cout
        << "***********************************************************************************\n";
    std::cout << "    OUTER ITERATION STEP    \n";
    std::cout
        << "***********************************************************************************\n";
    printf("+--------------+------------------------+--------------------+--------------------+\n");
    printf(
        "|-  step/max  -|-  tol      [norm]     -|--  fluid-inc      --|--  disp-inc      --|\n");
    printf("|   %3d/%3d    |  %10.3E[L_2 ]      | %10.3E         | %10.3E         |", itnum, itmax_,
        ittol_, fluidincnorm_L2 / fluidnorm_L2, dispincnorm_L2 / structnorm_L2);
    printf("\n");
    printf("+--------------+------------------------+--------------------+--------------------+\n");
  }

  // converged
  if ((fluidincnorm_L2 / fluidnorm_L2 <= ittol_) && (dispincnorm_L2 / structnorm_L2 <= ittol_))
  {
    stopnonliniter = true;
    if (get_comm().MyPID() == 0)
    {
      printf("\n");
      printf(
          "|  Outer Iteration loop converged after iteration %3d/%3d !                       |\n",
          itnum, itmax_);
      printf(
          "+--------------+------------------------+--------------------+--------------------+\n");
    }
  }

  // warn if itemax is reached without convergence, but proceed to next
  // timestep
  if ((itnum == itmax_) and
      ((fluidincnorm_L2 / fluidnorm_L2 > ittol_) || (dispincnorm_L2 / structnorm_L2 > ittol_)))
  {
    stopnonliniter = true;
    if ((get_comm().MyPID() == 0))  // and print_screen_evry() and (Step()%print_screen_evry()==0))
    {
      printf(
          "|     >>>>>> not converged in itemax steps!                                       |\n");
      printf(
          "+--------------+------------------------+--------------------+--------------------+\n");
      printf("\n");
      printf("\n");
    }
  }

  return stopnonliniter;
}

Teuchos::RCP<const Epetra_Map> PoroElast::Partitioned::dof_row_map_structure()
{
  return structure_field()->dof_row_map();
}

Teuchos::RCP<const Epetra_Map> PoroElast::Partitioned::dof_row_map_fluid()
{
  return fluid_field()->dof_row_map();
}

FOUR_C_NAMESPACE_CLOSE
