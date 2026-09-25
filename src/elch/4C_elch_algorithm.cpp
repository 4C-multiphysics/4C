// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_elch_algorithm.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_linalg_mapextractor.hpp"
#include "4C_scatra_timint_elch.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
ElCh::Algorithm::Algorithm(Global::Problem& problem, MPI_Comm comm,
    const Teuchos::ParameterList& elchcontrol, const Teuchos::ParameterList& scatradyn,
    const Teuchos::ParameterList& fdyn, const Teuchos::ParameterList& solverparams)
    : ScaTraAlgorithm(problem, comm, scatradyn, fdyn, "scatra", solverparams)
{
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ElCh::Algorithm::prepare_time_loop()
{
  // provide information about initial field (do not do for restarts!)
  if (step() == 0)
  {
    scatra_field()->output_total_and_mean_scalars();

    // compute error for problems with analytical solution (initial field!)
    scatra_field()->evaluate_error_compared_to_analytical_sol();
  }
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ElCh::Algorithm::print_scatra_solver()
{
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
    std::cout
        << "\n****************************\n      ELCH SOLVER\n****************************\n";
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
bool ElCh::Algorithm::convergence_check(
    int natconvitnum, const int natconvitmax, const double natconvittol)
{
  // convergence check based on the concentration, potential and
  // velocity increment

  //   | concentration increment |_2
  //  -------------------------------- < Tolerance
  //     | concentration_n+1 |_2

  bool stopnonliniter = false;
  std::shared_ptr<Core::LinAlg::MapExtractor> conpotsplitter = scatra_field()->splitter();
  // Variables to save different L2 - Norms

  double potincnorm_L2 = 0.0;
  double potnorm_L2 = 0.0;
  double velincnorm_L2 = 0.0;
  double velnorm_L2 = 0.0;
  double connorm_L2 = 0.0;
  double conincnorm_L2 = 0.0;

  // Calculate velocity increment and velocity L2 - Norm
  // velincnp_ = 1.0 * convelnp_ - 1.0 * conveln_

  velincnp_->update(1.0, *fluid_field()->extract_velocity_part(fluid_field()->velnp()), -1.0);
  velincnp_->norm_2(&velincnorm_L2);  // Estimation of the L2 - norm save values to both variables
                                      // (velincnorm_L2 and velnorm_L2)
  fluid_field()->extract_velocity_part(fluid_field()->velnp())->norm_2(&velnorm_L2);

  // Calculate concentration increment and concentration L2 - Norm
  phiincnp_->update(1.0, *scatra_field()->phinp(), -1.0);
  auto onlycon = conpotsplitter->extract_other_vector(*phiincnp_);
  onlycon->norm_2(&conincnorm_L2);
  conpotsplitter->extract_other_vector(*scatra_field()->phinp(), *onlycon);
  onlycon->norm_2(&connorm_L2);

  // Calculate potential increment and potential L2 - Norm
  auto onlypot = conpotsplitter->extract_cond_vector(*phiincnp_);
  onlypot->norm_2(&potincnorm_L2);
  conpotsplitter->extract_cond_vector(*scatra_field()->phinp(), *onlypot);
  onlypot->norm_2(&potnorm_L2);

  // care for the case that there is (almost) zero temperature or velocity
  // (usually not required for temperature)
  if (velnorm_L2 < 1e-6) velnorm_L2 = 1.0;
  if (connorm_L2 < 1e-6) connorm_L2 = 1.0;
  if (potnorm_L2 < 1e-6) potnorm_L2 = 1.0;

  // Print the incremental based convergence check to the screen
  if (natconvitnum != 1)
  {
    if (Core::Communication::my_mpi_rank(get_comm()) == 0)
    {
      std::cout << "\n";
      std::cout
          << "*****************************************************************************\n";
      std::cout << "                          OUTER ITERATION STEP\n";
      std::cout
          << "*****************************************************************************\n";
      printf("+------------+-------------------+--------------+-------------+-------------+\n");
      printf("|- step/max -|- tol      [norm] -|-- con-inc ---|-- pot-inc --|-- vel-inc --|\n");
      printf("|  %3d/%3d   | %10.3E[L_2 ]  | %10.3E   | %10.3E  | %10.3E  |", natconvitnum,
          natconvitmax, natconvittol, conincnorm_L2 / connorm_L2, potincnorm_L2 / potnorm_L2,
          velincnorm_L2 / velnorm_L2);
      printf("\n");
      printf("+------------+-------------------+--------------+-------------+-------------+\n");
    }

    // Converged or Not
    if ((conincnorm_L2 / connorm_L2 <= natconvittol) &&
        (potincnorm_L2 / potnorm_L2 <= natconvittol) &&
        (velincnorm_L2 / velnorm_L2 <= natconvittol))
    // if ((incconnorm_L2/connorm_L2 <= natconvittol))
    {
      stopnonliniter = true;
      if (Core::Communication::my_mpi_rank(get_comm()) == 0)
      {
        printf("| Outer Iteration loop converged after iteration %3d/%3d                    |\n",
            natconvitnum, natconvitmax);
        printf("+---------------------------------------------------------------------------+");
        printf("\n");
        printf("\n");
      }
    }
    else
    {
      if (Core::Communication::my_mpi_rank(get_comm()) == 0)
      {
        printf("| Outer Iteration loop is not converged after iteration %3d/%3d             |\n",
            natconvitnum, natconvitmax);
        printf("+---------------------------------------------------------------------------+");
        printf("\n");
        printf("\n");
      }
    }
  }
  else
  {
    // first outer iteration loop: fluid solver has not got the new density yet
    // => minimum two outer iteration loops
    stopnonliniter = false;
    if (Core::Communication::my_mpi_rank(get_comm()) == 0)
    {
      std::cout << "\n";
      std::cout
          << "*****************************************************************************\n";
      std::cout << "                          OUTER ITERATION STEP\n";
      std::cout
          << "*****************************************************************************\n";
      printf("+------------+-------------------+--------------+-------------+-------------+\n");
      printf("|- step/max -|- tol      [norm] -|-- con-inc ---|-- pot-inc --|-- vel-inc --|\n");
      printf("|  %3d/%3d   | %10.3E[L_2 ]  |       -      |      -      |      -      |",
          natconvitnum, natconvitmax, natconvittol);
      printf("\n");
      printf("+------------+-------------------+--------------+-------------+-------------+\n");
    }
  }

  // warn if itemax is reached without convergence, but proceed to next timestep
  // itemax = 1 is also possible for segregated coupling approaches (not fully implicit)
  if (natconvitnum == natconvitmax)
  {
    if (((conincnorm_L2 / connorm_L2 > natconvittol) ||
            (potincnorm_L2 / potnorm_L2 > natconvittol) ||
            (velincnorm_L2 / velnorm_L2 > natconvittol)) or
        (natconvitmax == 1))
    {
      stopnonliniter = true;
      if ((Core::Communication::my_mpi_rank(get_comm()) == 0))
      {
        printf("|     >>>>>> not converged in itemax steps!     |\n");
        printf("+-----------------------------------------------+\n");
        printf("\n");
        printf("\n");
      }
    }
  }

  return stopnonliniter;
}

FOUR_C_NAMESPACE_CLOSE
