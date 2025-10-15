// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_sti_partitioned.hpp"

#include "4C_adapter_scatra_base_algorithm.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_scatra_timint_implicit.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN

/*--------------------------------------------------------------------------------*
 | constructor                                                         fang 09/17 |
 *--------------------------------------------------------------------------------*/
STI::Partitioned::Partitioned(MPI_Comm comm,  //! communicator
    const Teuchos::ParameterList& stidyn,     //! parameter list for scatra-thermo interaction
    const Teuchos::ParameterList&
        scatradyn,  //! scalar transport parameter list for scatra and thermo fields
    const Teuchos::ParameterList& solverparams_scatra,  //! solver parameter list for scatra field
    const Teuchos::ParameterList& solverparams_thermo   //! solver parameter list for thermo field
    )
    :  // instantiate base class
      Algorithm(comm, stidyn, scatradyn, solverparams_scatra, solverparams_thermo),
      couplingtype_(Teuchos::getIntegralValue<STI::CouplingType>(stidyn, "COUPLINGTYPE")),
      omegamax_(stidyn.sublist("PARTITIONED").get<double>("OMEGAMAX"))
{
  // set control parameters for outer coupling iteration loop
  itermax_ = fieldparameters_->sublist("NONLINEAR").get<int>("ITEMAX_OUTER");
  itertol_ = fieldparameters_->sublist("NONLINEAR").get<double>("CONVTOL_OUTER");

  // initialize vectors for outer coupling iteration loop
  switch (couplingtype_)
  {
    case STI::CouplingType::oneway_scatratothermo:
    case STI::CouplingType::oneway_thermotoscatra:
      // do nothing
      break;

    case STI::CouplingType::twoway_scatratothermo:
    case STI::CouplingType::twoway_scatratothermo_aitken:
    case STI::CouplingType::twoway_scatratothermo_aitken_dofsplit:
    case STI::CouplingType::twoway_thermotoscatra:
    case STI::CouplingType::twoway_thermotoscatra_aitken:
    {
      // initialize increment vectors
      scatra_field()->set_phinp_inc(
          Core::LinAlg::create_vector(*scatra_field()->discretization()->dof_row_map(), true));
      thermo_field()->set_phinp_inc(
          Core::LinAlg::create_vector(*thermo_field()->discretization()->dof_row_map(), true));

      // initialize old increment vectors
      if (couplingtype_ == STI::CouplingType::twoway_scatratothermo_aitken or
          couplingtype_ == STI::CouplingType::twoway_scatratothermo_aitken_dofsplit)
      {
        scatra_field()->set_phinp_inc_old(
            Core::LinAlg::create_vector(*scatra_field()->discretization()->dof_row_map(), true));
      }
      else if (couplingtype_ == STI::CouplingType::twoway_thermotoscatra_aitken)
      {
        thermo_field()->set_phinp_inc_old(
            Core::LinAlg::create_vector(*thermo_field()->discretization()->dof_row_map(), true));
      }

      // initialize relaxation parameter
      if (couplingtype_ == STI::CouplingType::twoway_scatratothermo)
        scatra_field()->omega().resize(1, stidyn.sublist("PARTITIONED").get<double>("OMEGA"));
      else if (couplingtype_ == STI::CouplingType::twoway_scatratothermo_aitken)
        scatra_field()->omega().resize(1, 1.);
      else if (couplingtype_ == STI::CouplingType::twoway_scatratothermo_aitken_dofsplit)
        scatra_field()->omega().resize(scatra_field()->num_dof_per_node(), 1.);
      else if (couplingtype_ == STI::CouplingType::twoway_thermotoscatra)
        thermo_field()->omega().resize(1, stidyn.sublist("PARTITIONED").get<double>("OMEGA"));
      else if (couplingtype_ == STI::CouplingType::twoway_thermotoscatra_aitken)
        thermo_field()->omega().resize(1, 1.);
      else
        thermo_field()->omega().resize(scatra_field()->num_dof_per_node(), 1.);

      break;
    }

    default:
    {
      FOUR_C_THROW("Invalid coupling type. You should not end up here!");
    }
  }

}  // STI::Partitioned::Partitioned


/*--------------------------------------------------------------------*
 | convergence check for outer iteration loop              fang 09/17 |
 *--------------------------------------------------------------------*/
bool STI::Partitioned::exit_outer_coupling() const
{
  // extract processor ID
  const int mypid = Core::Communication::my_mpi_rank(get_comm());

  // compute vector norms
  double L2_scatra(0.), L2_scatra_inc(0.), L2_thermo(0.), L2_thermo_inc(0.);
  scatra_field()->phinp()->norm_2(&L2_scatra);
  scatra_field()->phinp_inc()->norm_2(&L2_scatra_inc);
  thermo_field()->phinp()->norm_2(&L2_thermo);
  thermo_field()->phinp_inc()->norm_2(&L2_thermo_inc);
  if (L2_scatra < 1.e-10) L2_scatra = 1.;
  if (L2_thermo < 1.e-10) L2_thermo = 1.;

  // print convergence status
  if (mypid == 0)
  {
    std::cout << std::endl;
    std::cout << "+------------+-------------------+--------------+--------------+" << std::endl;
    std::cout << "|                       OUTER ITERATION                        |" << std::endl;
    std::cout << "+------------+-------------------+--------------+--------------+" << std::endl;
    std::cout << "|- step/max -|- tol      [norm] -|- scatra-inc -|- thermo-inc -|" << std::endl;
    std::cout << "|  " << std::setw(3) << iter_ << "/" << std::setw(3) << itermax_ << "   | "
              << std::setw(10) << std::setprecision(3) << std::scientific << itertol_
              << "[L_2 ]  | " << std::setw(10) << std::setprecision(3) << std::scientific
              << L2_scatra_inc / L2_scatra << "   | " << std::setw(10) << std::setprecision(3)
              << std::scientific << L2_thermo_inc / L2_thermo << "   |" << std::endl;
    std::cout << "+------------+-------------------+--------------+--------------+" << std::endl;
  }

  // convergence check
  if (L2_scatra_inc / L2_scatra <= itertol_ and L2_thermo_inc / L2_thermo <= itertol_)
  {
    if (mypid == 0)
    {
      std::cout << "|   OUTER ITERATION LOOP CONVERGED AFTER ITERATION " << std::setw(3) << iter_
                << "/" << std::setw(3) << itermax_ << " !   |" << std::endl;
      std::cout << "+------------+-------------------+--------------+--------------+" << std::endl;
    }

    return true;
  }

  // throw error in case maximum number of iteration steps is reached without convergence
  else if (iter_ == itermax_)
  {
    if (mypid == 0)
    {
      std::cout << "| >>>> not converged within maximum number of iteration steps! |" << std::endl;
      std::cout << "+------------+-------------------+--------------+--------------+" << std::endl;
    }

    FOUR_C_THROW("Outer iteration did not converge within maximum number of iteration steps!");

    return true;
  }

  // proceed with next outer iteration step
  return false;
}  // STI::Partitioned::exit_outer_coupling()


/*--------------------------------------------------------------------------------*
 | evaluate time step using outer coupling iteration                   fang 09/17 |
 *--------------------------------------------------------------------------------*/
void STI::Partitioned::solve()
{
  switch (couplingtype_)
  {
    case STI::CouplingType::oneway_scatratothermo:
    case STI::CouplingType::oneway_thermotoscatra:
    {
      solve_one_way();
      break;
    }

    case STI::CouplingType::twoway_scatratothermo:
    case STI::CouplingType::twoway_scatratothermo_aitken:
    case STI::CouplingType::twoway_scatratothermo_aitken_dofsplit:
    case STI::CouplingType::twoway_thermotoscatra:
    case STI::CouplingType::twoway_thermotoscatra_aitken:
    {
      solve_two_way();
      break;
    }

    default:
    {
      FOUR_C_THROW("Invalid coupling type. You should not end up here!");
      break;
    }
  }
}  // STI::Partitioned::Solve()


/*--------------------------------------------------------------------------------*
 | evaluate time step using one-way coupling iteration                 fang 09/17 |
 *--------------------------------------------------------------------------------*/
void STI::Partitioned::solve_one_way() const
{
  switch (couplingtype_)
  {
    case STI::CouplingType::oneway_scatratothermo:
    {
      // pass thermo degrees of freedom to scatra discretization
      transfer_thermo_to_scatra(thermo_field()->phiafnp());

      // solve scatra field
      scatra_field()->solve();

      // pass scatra degrees of freedom to thermo discretization
      transfer_scatra_to_thermo(scatra_field()->phiafnp());

      // solve thermo field
      thermo_field()->solve();

      break;
    }

    case STI::CouplingType::oneway_thermotoscatra:
    {
      // pass scatra degrees of freedom to thermo discretization
      transfer_scatra_to_thermo(scatra_field()->phiafnp());

      // solve thermo field
      thermo_field()->solve();

      // pass thermo degrees of freedom to scatra discretization
      transfer_thermo_to_scatra(thermo_field()->phiafnp());

      // solve scatra field
      scatra_field()->solve();

      break;
    }

    default:
    {
      FOUR_C_THROW("Invalid coupling type. You should not end up here!");
      break;
    }
  }
}  // STI::Partitioned::solve_one_way()


/*----------------------------------------------------------------------*
 | evaluate time step using two-way coupling iteration       fang 09/17 |
 *----------------------------------------------------------------------*/
void STI::Partitioned::solve_two_way()
{
  // reset number of outer iterations
  iter_ = 0;

  switch (couplingtype_)
  {
    case STI::CouplingType::twoway_scatratothermo:
    case STI::CouplingType::twoway_scatratothermo_aitken:
    case STI::CouplingType::twoway_scatratothermo_aitken_dofsplit:
    {
      // initialize relaxed scatra state vector
      const std::shared_ptr<Core::LinAlg::Vector<double>> scatra_relaxed =
          std::make_shared<Core::LinAlg::Vector<double>>(*scatra_field()->phiafnp());

      // begin outer iteration loop
      while (true)
      {
        // increment iteration number
        iter_++;

        // pass relaxed scatra degrees of freedom to thermo discretization
        transfer_scatra_to_thermo(scatra_relaxed);

        // store current thermo state vector
        thermo_field()->phinp_inc()->update(1., *thermo_field()->phiafnp(), 0.);

        // solve thermo field
        thermo_field()->solve();

        // compute increment of thermo state vector
        thermo_field()->phinp_inc()->update(1., *thermo_field()->phiafnp(), -1.);

        // pass thermo degrees of freedom to scatra discretization
        transfer_thermo_to_scatra(thermo_field()->phiafnp());

        // store current scatra state vector
        scatra_field()->phinp_inc()->update(1., *scatra_relaxed, 0.);

        // solve scatra field
        scatra_field()->solve();

        // compute increment of scatra state vector
        scatra_field()->phinp_inc()->update(1., *scatra_field()->phiafnp(), -1.);

        // convergence check
        if (exit_outer_coupling()) break;

        // perform static relaxation
        if (couplingtype_ == STI::CouplingType::twoway_scatratothermo)
        {
          scatra_relaxed->update(scatra_field()->omega()[0], *scatra_field()->phinp_inc(), 1.);
        }

        // perform dynamic relaxation
        else
        {
          // compute difference between current and previous increments of scatra state vector
          Core::LinAlg::Vector<double> scatra_inc_diff(*scatra_field()->phinp_inc());
          scatra_inc_diff.update(-1., *scatra_field()->phinp_inc_old(), 1.);

          if (couplingtype_ == STI::CouplingType::twoway_scatratothermo_aitken)
          {
            // compute L2 norm of difference between current and previous increments of scatra state
            // vector
            double scatra_inc_diff_L2(0.);
            scatra_inc_diff.norm_2(&scatra_inc_diff_L2);

            // compute dot product between increment of scatra state vector and difference between
            // current and previous increments of scatra state vector
            double scatra_inc_dot_scatra_inc_diff(0.);
            scatra_inc_diff.dot(*scatra_field()->phinp_inc(), &scatra_inc_dot_scatra_inc_diff);

            // compute Aitken relaxation factor
            if (iter_ > 1 and scatra_inc_diff_L2 > 1.e-12)
              scatra_field()->omega()[0] *=
                  1 - scatra_inc_dot_scatra_inc_diff / (scatra_inc_diff_L2 * scatra_inc_diff_L2);

            // restrict Aitken relaxation factor if necessary
            if (omegamax_ > 0. and scatra_field()->omega()[0] > omegamax_)
              scatra_field()->omega()[0] = omegamax_;

            // perform Aitken relaxation
            scatra_relaxed->update(scatra_field()->omega()[0], *scatra_field()->phinp_inc(), 1.);
          }

          else
          {
            // safety check
            if (scatra_field()->splitter() == nullptr)
              FOUR_C_THROW("Map extractor was not initialized!");

            // loop over all degrees of freedom
            for (int idof = 0; idof < scatra_field()->splitter()->num_maps(); ++idof)
            {
              // extract subvectors associated with current degree of freedom
              const std::shared_ptr<const Core::LinAlg::Vector<double>> scatra_inc_dof =
                  scatra_field()->splitter()->extract_vector(*scatra_field()->phinp_inc(), idof);
              const std::shared_ptr<const Core::LinAlg::Vector<double>> scatra_inc_diff_dof =
                  scatra_field()->splitter()->extract_vector(scatra_inc_diff, idof);

              // compute L2 norm of difference between current and previous increments of current
              // degree of freedom
              double scatra_inc_diff_L2(0.);
              scatra_inc_diff_dof->norm_2(&scatra_inc_diff_L2);

              // compute dot product between increment of current degree of freedom and difference
              // between current and previous increments of current degree of freedom
              double scatra_inc_dot_scatra_inc_diff(0.);
              scatra_inc_diff_dof->dot(*scatra_inc_dof, &scatra_inc_dot_scatra_inc_diff);

              // compute Aitken relaxation factor for current degree of freedom
              if (iter_ > 1 and scatra_inc_diff_L2 > 1.e-12)
                scatra_field()->omega()[idof] *=
                    1 - scatra_inc_dot_scatra_inc_diff / (scatra_inc_diff_L2 * scatra_inc_diff_L2);

              // restrict Aitken relaxation factor if necessary
              if (omegamax_ > 0. and scatra_field()->omega()[idof] > omegamax_)
                scatra_field()->omega()[idof] = omegamax_;

              // perform Aitken relaxation for current degree of freedom
              scatra_field()->splitter()->add_vector(
                  *scatra_inc_dof, idof, *scatra_relaxed, scatra_field()->omega()[idof]);
            }
          }

          // update increment of scatra state vector
          scatra_field()->phinp_inc_old()->update(1., *scatra_field()->phinp_inc(), 0.);
        }
      }

      break;
    }

    case STI::CouplingType::twoway_thermotoscatra:
    case STI::CouplingType::twoway_thermotoscatra_aitken:
    {
      // initialize relaxed thermo state vector
      const std::shared_ptr<Core::LinAlg::Vector<double>> thermo_relaxed =
          std::make_shared<Core::LinAlg::Vector<double>>(*thermo_field()->phiafnp());

      // begin outer iteration loop
      while (true)
      {
        // increment iteration number
        iter_++;

        // pass relaxed thermo degrees of freedom to scatra discretization
        transfer_thermo_to_scatra(thermo_relaxed);

        // store current scatra state vector
        scatra_field()->phinp_inc()->update(1., *scatra_field()->phiafnp(), 0.);

        // solve scatra field
        scatra_field()->solve();

        // compute increment of scatra state vector
        scatra_field()->phinp_inc()->update(1., *scatra_field()->phiafnp(), -1.);

        // pass scatra degrees of freedom to thermo discretization
        transfer_scatra_to_thermo(scatra_field()->phiafnp());

        // store current thermo state vector
        thermo_field()->phinp_inc()->update(1., *thermo_relaxed, 0.);

        // solve thermo field
        thermo_field()->solve();

        // compute increment of thermo state vector
        thermo_field()->phinp_inc()->update(1., *thermo_field()->phiafnp(), -1.);

        // convergence check
        if (exit_outer_coupling()) break;

        if (couplingtype_ == STI::CouplingType::twoway_thermotoscatra_aitken)
        {
          // compute difference between current and previous increments of thermo state vector
          Core::LinAlg::Vector<double> thermo_inc_diff(*thermo_field()->phinp_inc());
          thermo_inc_diff.update(-1., *thermo_field()->phinp_inc_old(), 1.);

          // compute L2 norm of difference between current and previous increments of thermo state
          // vector
          double thermo_inc_diff_L2(0.);
          thermo_inc_diff.norm_2(&thermo_inc_diff_L2);

          // compute dot product between increment of thermo state vector and difference between
          // current and previous increments of thermo state vector
          double thermo_inc_dot_thermo_inc_diff(0.);
          thermo_inc_diff.dot(*thermo_field()->phinp_inc(), &thermo_inc_dot_thermo_inc_diff);

          // compute Aitken relaxation factor
          if (iter_ > 1 and thermo_inc_diff_L2 > 1.e-12)
            thermo_field()->omega()[0] *=
                1 - thermo_inc_dot_thermo_inc_diff / (thermo_inc_diff_L2 * thermo_inc_diff_L2);

          // restrict Aitken relaxation factor if necessary
          if (omegamax_ > 0. and thermo_field()->omega()[0] > omegamax_)
            thermo_field()->omega()[0] = omegamax_;

          // update increment of thermo state vector
          thermo_field()->phinp_inc_old()->update(1., *thermo_field()->phinp_inc(), 0.);
        }

        // perform relaxation
        thermo_relaxed->update(thermo_field()->omega()[0], *thermo_field()->phinp_inc(), 1.);
      }

      break;
    }

    default:
    {
      FOUR_C_THROW("Invalid coupling type. You should not end up here!");
      break;
    }
  }
}  // STI::Partitioned::solve_two_way()

FOUR_C_NAMESPACE_CLOSE
