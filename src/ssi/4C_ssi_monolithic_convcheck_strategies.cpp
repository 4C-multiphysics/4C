// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_ssi_monolithic_convcheck_strategies.hpp"

#include "4C_adapter_scatra_base_algorithm.hpp"
#include "4C_adapter_str_ssiwrapper.hpp"
#include "4C_comm_mpi_utils.hpp"
#include "4C_linalg_mapextractor.hpp"
#include "4C_scatra_timint_implicit.hpp"
#include "4C_ssi_utils.hpp"

#include <Teuchos_ParameterList.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
SSI::SsiMono::ConvCheckStrategyBase::ConvCheckStrategyBase(const Teuchos::ParameterList& parameters)
    : itermax_(parameters.get<int>("ITEMAX")),
      itertol_(parameters.sublist("MONOLITHIC").get<double>("CONVTOL")),
      restol_(parameters.sublist("MONOLITHIC").get<double>("ABSTOLRES"))
{
}

/*-----------------------------------------------------------------------*
 *-----------------------------------------------------------------------*/
bool SSI::SsiMono::ConvCheckStrategyBase::exit_newton_raphson(const SSI::SsiMono& ssi_mono)
{
  const auto norms = compute_norms(ssi_mono);
  const bool converged = check_convergence(ssi_mono, norms);
  const bool exit = compute_exit(ssi_mono, converged);

  print_newton_iteration_information(ssi_mono, converged, exit, norms);

  if (exit and !converged) non_converged_steps_.insert(ssi_mono.step());

  return exit;
}

/*-----------------------------------------------------------------------*
 *-----------------------------------------------------------------------*/
bool SSI::SsiMono::ConvCheckStrategyBase::compute_exit(
    const SSI::SsiMono& ssi_mono, const bool converged) const
{
  return (converged or ssi_mono.iteration_count() == itermax_);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSI::SsiMono::ConvCheckStrategyBase::check_l2_norm(
    const double& incnorm, const double& resnorm, double& dofnorm) const
{
  if (std::isnan(incnorm) or std::isnan(resnorm) or std::isnan(dofnorm))
    FOUR_C_THROW("Vector norm is not a number!");
  if (std::isinf(incnorm) or std::isinf(resnorm) or std::isinf(dofnorm))
    FOUR_C_THROW("Vector norm is infinity!");

  if (dofnorm < 1.e-10) dofnorm = 1.e-10;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSI::SsiMono::ConvCheckStrategyBase::get_and_check_l2_norm_structure(
    const SSI::SsiMono& ssi_mono, double& incnorm, double& resnorm, double& dofnorm) const
{
  ssi_mono.maps_sub_problems()
      ->extract_vector(*ssi_mono.ssi_vectors_->increment(),
          Utils::SSIMaps::get_problem_position(Subproblem::structure))
      ->norm_2(&incnorm);

  ssi_mono.maps_sub_problems()
      ->extract_vector(*ssi_mono.ssi_vectors_->residual(),
          Utils::SSIMaps::get_problem_position(Subproblem::structure))
      ->norm_2(&resnorm);

  ssi_mono.structure_field()->dispnp()->norm_2(&dofnorm);

  check_l2_norm(incnorm, resnorm, dofnorm);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SsiMono::ConvCheckStrategyBase::print_non_converged_steps(const int pid) const
{
  if (pid == 0 and not non_converged_steps_.empty())
  {
    std::cout << "\nNon converged time steps: ";
    for (auto it = non_converged_steps_.begin(); it != non_converged_steps_.end(); ++it)
    {
      std::cout << *it;
      // print comma except after the last element
      if (std::next(it) != non_converged_steps_.end()) std::cout << ", ";
    }
    std::cout << "\n\n";
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSI::SsiMono::ConvCheckStrategyStd::get_and_check_l2_norm_scatra(
    const SSI::SsiMono& ssi_mono, double& incnorm, double& resnorm, double& dofnorm) const
{
  ssi_mono.maps_sub_problems()
      ->extract_vector(*ssi_mono.ssi_vectors_->increment(),
          Utils::SSIMaps::get_problem_position(Subproblem::scalar_transport))
      ->norm_2(&incnorm);

  ssi_mono.maps_sub_problems()
      ->extract_vector(*ssi_mono.ssi_vectors_->residual(),
          Utils::SSIMaps::get_problem_position(Subproblem::scalar_transport))
      ->norm_2(&resnorm);

  ssi_mono.scatra_field()->phinp()->norm_2(&dofnorm);

  check_l2_norm(incnorm, resnorm, dofnorm);
}

/*-----------------------------------------------------------------------*
 *-----------------------------------------------------------------------*/
std::map<SSI::L2norm, double> SSI::SsiMono::ConvCheckStrategyStd::compute_norms(
    const SSI::SsiMono& ssi_mono) const
{
  double scatraincnorm = 0.0, scatraresnorm = 0.0, scatradofnorm = 0.0, structureincnorm = 0.0,
         structureresnorm = 0.0, structuredofnorm = 0.0;

  get_and_check_l2_norm_scatra(ssi_mono, scatraincnorm, scatraresnorm, scatradofnorm);
  get_and_check_l2_norm_structure(ssi_mono, structureincnorm, structureresnorm, structuredofnorm);

  return {{SSI::L2norm::scatraincnorm, scatraincnorm}, {SSI::L2norm::scatraresnorm, scatraresnorm},
      {SSI::L2norm::scatradofnorm, scatradofnorm},
      {SSI::L2norm::structureincnorm, structureincnorm},
      {SSI::L2norm::structureresnorm, structureresnorm},
      {SSI::L2norm::structuredofnorm, structuredofnorm}};
}

/*-----------------------------------------------------------------------*
 *-----------------------------------------------------------------------*/
bool SSI::SsiMono::ConvCheckStrategyStd::check_convergence(
    const SSI::SsiMono& ssi_mono, const std::map<L2norm, double>& norms) const
{
  if (ssi_mono.iteration_count() > 1 and norms.at(L2norm::scatraresnorm) <= itertol_ and
      norms.at(L2norm::structureresnorm) <= itertol_ and
      norms.at(L2norm::scatraincnorm) / norms.at(L2norm::scatradofnorm) <= itertol_ and
      norms.at(L2norm::structureincnorm) / norms.at(L2norm::structuredofnorm) <= itertol_)
    return true;

  if (norms.at(L2norm::scatraresnorm) < restol_ and norms.at(L2norm::structureresnorm) < restol_)
    return true;

  return false;
}

/*-----------------------------------------------------------------------*
 *-----------------------------------------------------------------------*/
void SSI::SsiMono::ConvCheckStrategyStd::print_newton_iteration_information(
    const SSI::SsiMono& ssi_mono, const bool converged, const bool exit,
    const std::map<L2norm, double>& norms) const
{
  if (Core::Communication::my_mpi_rank(ssi_mono.get_comm()) != 0) return;

  if (ssi_mono.iteration_count() == 1)
  {
    // print header of convergence table to screen
    std::cout << "+------------+-------------------+--------------+--------------+--------------+"
                 "--------------+\n";
    std::cout << "|- step/max -|- tolerance[norm] -|- scatra-res -|- scatra-inc -|- struct-res "
                 "-|- struct-inc -|\n";

    // print first line of convergence table to screen
    // solution increment not yet available during first Newton-Raphson iteration
    std::cout << "|  " << std::setw(3) << ssi_mono.iteration_count() << "/" << std::setw(3)
              << itermax_ << "   | " << std::setw(10) << std::setprecision(3) << std::scientific
              << itertol_ << "[L_2 ]  | " << std::setw(10) << std::setprecision(3)
              << std::scientific << norms.at(L2norm::scatraresnorm) << "   |      --      | "
              << std::setw(10) << std::setprecision(3) << std::scientific
              << norms.at(L2norm::structureresnorm) << "   |      --      | "
              << "(       --      , te = " << std::setw(10) << std::setprecision(3)
              << ssi_mono.dt_eval_ << ")\n";
  }
  else
  {
    std::cout << "|  " << std::setw(3) << ssi_mono.iteration_count() << "/" << std::setw(3)
              << itermax_ << "   | " << std::setw(10) << std::setprecision(3) << std::scientific
              << itertol_ << "[L_2 ]  | " << std::setw(10) << std::setprecision(3)
              << std::scientific << norms.at(L2norm::scatraresnorm) << "   | " << std::setw(10)
              << std::setprecision(3) << std::scientific
              << norms.at(L2norm::scatraincnorm) / norms.at(L2norm::scatradofnorm) << "   | "
              << std::setw(10) << std::setprecision(3) << std::scientific
              << norms.at(L2norm::structureresnorm) << "   | " << std::setw(10)
              << std::setprecision(3) << std::scientific
              << norms.at(L2norm::structureincnorm) / norms.at(L2norm::structuredofnorm)
              << "   | (ts = " << std::setw(10) << std::setprecision(3) << ssi_mono.dt_solve_
              << ", te = " << std::setw(10) << std::setprecision(3) << ssi_mono.dt_eval_ << ")\n";
  }

  if (exit and !converged)
  {
    std::cout << "+------------+-------------------+--------------+--------------+--------------+"
                 "--------------+\n";
    std::cout << "|      Newton-Raphson method has not converged after a maximum number of "
              << std::setw(2) << itermax_ << " iterations!      |\n";
    std::cout << "+------------+-------------------+--------------+--------------+--------------+--"
                 "------------+\n";
  }

  if (exit and converged)
  {
    std::cout << "+------------+-------------------+--------------+--------------+--------------+--"
                 "------------+\n";
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSI::SsiMono::ConvCheckStrategyElch::get_and_check_l2_norm_conc(
    const SSI::SsiMono& ssi_mono, double& incnorm, double& resnorm, double& dofnorm) const
{
  ssi_mono.scatra_field()
      ->splitter()
      ->extract_other_vector(
          *ssi_mono.maps_sub_problems()->extract_vector(*ssi_mono.ssi_vectors_->increment(),
              Utils::SSIMaps::get_problem_position(Subproblem::scalar_transport)))
      ->norm_2(&incnorm);

  ssi_mono.scatra_field()
      ->splitter()
      ->extract_other_vector(
          *ssi_mono.maps_sub_problems()->extract_vector(*ssi_mono.ssi_vectors_->residual(),
              Utils::SSIMaps::get_problem_position(Subproblem::scalar_transport)))
      ->norm_2(&resnorm);

  ssi_mono.scatra_field()
      ->splitter()
      ->extract_other_vector(*ssi_mono.scatra_field()->phinp())
      ->norm_2(&dofnorm);

  check_l2_norm(incnorm, resnorm, dofnorm);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSI::SsiMono::ConvCheckStrategyElch::get_and_check_l2_norm_pot(
    const SSI::SsiMono& ssi_mono, double& incnorm, double& resnorm, double& dofnorm) const
{
  ssi_mono.scatra_field()
      ->splitter()
      ->extract_cond_vector(
          *ssi_mono.maps_sub_problems()->extract_vector(*ssi_mono.ssi_vectors_->increment(),
              Utils::SSIMaps::get_problem_position(Subproblem::scalar_transport)))
      ->norm_2(&incnorm);

  ssi_mono.scatra_field()
      ->splitter()
      ->extract_cond_vector(
          *ssi_mono.maps_sub_problems()->extract_vector(*ssi_mono.ssi_vectors_->residual(),
              Utils::SSIMaps::get_problem_position(Subproblem::scalar_transport)))
      ->norm_2(&resnorm);

  ssi_mono.scatra_field()
      ->splitter()
      ->extract_cond_vector(*ssi_mono.scatra_field()->phinp())
      ->norm_2(&dofnorm);

  check_l2_norm(incnorm, resnorm, dofnorm);
}

/*-----------------------------------------------------------------------*
 *-----------------------------------------------------------------------*/
std::map<SSI::L2norm, double> SSI::SsiMono::ConvCheckStrategyElch::compute_norms(
    const SSI::SsiMono& ssi_mono) const
{
  double concincnorm = 0.0, concresnorm = 0.0, concdofnorm = 0.0, potdofnorm = 0.0,
         potincnorm = 0.0, potresnorm = 0.0, structuredofnorm = 0.0, structureresnorm = 0.0,
         structureincnorm = 0.0;

  get_and_check_l2_norm_conc(ssi_mono, concincnorm, concresnorm, concdofnorm);
  get_and_check_l2_norm_pot(ssi_mono, potincnorm, potresnorm, potdofnorm);
  get_and_check_l2_norm_structure(ssi_mono, structureincnorm, structureresnorm, structuredofnorm);

  return {
      {SSI::L2norm::concincnorm, concincnorm},
      {SSI::L2norm::concresnorm, concresnorm},
      {SSI::L2norm::concdofnorm, concdofnorm},
      {SSI::L2norm::potdofnorm, potdofnorm},
      {SSI::L2norm::potincnorm, potincnorm},
      {SSI::L2norm::potresnorm, potresnorm},
      {SSI::L2norm::structuredofnorm, structuredofnorm},
      {SSI::L2norm::structureresnorm, structureresnorm},
      {SSI::L2norm::structureincnorm, structureincnorm},
  };
}

/*-----------------------------------------------------------------------*
 *-----------------------------------------------------------------------*/
bool SSI::SsiMono::ConvCheckStrategyElch::check_convergence(
    const SSI::SsiMono& ssi_mono, const std::map<L2norm, double>& norms) const
{
  if (ssi_mono.iteration_count() > 1 and norms.at(SSI::L2norm::concresnorm) <= itertol_ and
      norms.at(SSI::L2norm::potresnorm) <= itertol_ and
      norms.at(SSI::L2norm::structureresnorm) <= itertol_ and
      norms.at(SSI::L2norm::concincnorm) / norms.at(SSI::L2norm::concdofnorm) <= itertol_ and
      norms.at(SSI::L2norm::potincnorm) / norms.at(SSI::L2norm::potdofnorm) <= itertol_ and
      norms.at(SSI::L2norm::structureincnorm) / norms.at(SSI::L2norm::structuredofnorm) <= itertol_)
    return true;

  if (norms.at(SSI::L2norm::concresnorm) < restol_ and
      norms.at(SSI::L2norm::potresnorm) < restol_ and
      norms.at(SSI::L2norm::structureresnorm) < restol_)
    return true;

  return false;
}

/*-----------------------------------------------------------------------*
 *-----------------------------------------------------------------------*/
void SSI::SsiMono::ConvCheckStrategyElch::print_newton_iteration_information(
    const SSI::SsiMono& ssi_mono, const bool converged, const bool exit,
    const std::map<L2norm, double>& norms) const
{
  if (Core::Communication::my_mpi_rank(ssi_mono.get_comm()) != 0) return;

  if (ssi_mono.iteration_count() == 1)
  {
    // print header of convergence table to screen
    std::cout << "+------------+-------------------+--------------+--------------+--------------+"
                 "--------------+--------------+--------------+\n";
    std::cout << "|- step/max -|- tolerance[norm] -|-- conc-res --|-- conc-inc --|-- pot-res "
                 "---|-- pot-inc ---|- struct-res -|- struct-inc -|\n";

    // print first line of convergence table to screen
    // solution increment not yet available during first Newton-Raphson iteration
    std::cout << "|  " << std::setw(3) << ssi_mono.iteration_count() << "/" << std::setw(3)
              << itermax_ << "   | " << std::setw(10) << std::setprecision(3) << std::scientific
              << itertol_ << "[L_2 ]  | " << std::setw(10) << std::setprecision(3)
              << std::scientific << norms.at(SSI::L2norm::concresnorm) << "   |      --      | "
              << std::setw(10) << std::setprecision(3) << std::scientific
              << norms.at(SSI::L2norm::potresnorm) << "   |      --      | " << std::setw(10)
              << std::setprecision(3) << std::scientific << norms.at(SSI::L2norm::structureresnorm)
              << "   |      --      | "
              << "(       --      , te = " << std::setw(10) << std::setprecision(3)
              << ssi_mono.dt_eval_ << ")\n";
  }
  else
  {
    std::cout << "|  " << std::setw(3) << ssi_mono.iteration_count() << "/" << std::setw(3)
              << itermax_ << "   | " << std::setw(10) << std::setprecision(3) << std::scientific
              << itertol_ << "[L_2 ]  | " << std::setw(10) << std::setprecision(3)
              << std::scientific << norms.at(SSI::L2norm::concresnorm) << "   | " << std::setw(10)
              << std::setprecision(3) << std::scientific
              << norms.at(SSI::L2norm::concincnorm) / norms.at(SSI::L2norm::concdofnorm) << "   | "
              << std::setw(10) << std::setprecision(3) << std::scientific
              << norms.at(SSI::L2norm::potresnorm) << "   | " << std::setw(10)
              << std::setprecision(3) << std::scientific
              << norms.at(SSI::L2norm::potincnorm) / norms.at(SSI::L2norm::potdofnorm) << "   | "
              << std::setw(10) << std::setprecision(3) << std::scientific
              << norms.at(SSI::L2norm::structureresnorm) << "   | " << std::setw(10)
              << std::setprecision(3) << std::scientific
              << norms.at(SSI::L2norm::structureincnorm) / norms.at(SSI::L2norm::structuredofnorm)
              << "   | (ts = " << std::setw(10) << std::setprecision(3) << ssi_mono.dt_solve_
              << ", te = " << std::setw(10) << std::setprecision(3) << ssi_mono.dt_eval_ << ")\n";
  }

  if (exit and !converged)
  {
    std::cout << "+------------+-------------------+--------------+--------------+--------------+"
                 "--------------+--------------+--------------+\n";
    std::cout << "|                     Newton-Raphson method has not converged after a maximum "
                 "number of "
              << std::setw(2) << itermax_ << " iterations!                     |\n";
    std::cout << "+------------+-------------------+--------------+--------------+--------------+--"
                 "------------+--------------+--------------+\n";
  }

  if (exit and converged)
  {
    std::cout << "+------------+-------------------+--------------+--------------+--------------+--"
                 "------------+--------------+--------------+\n";
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool SSI::SsiMono::ConvCheckStrategyElch::exit_newton_raphson_init_pot_calc(
    const SSI::SsiMono& ssi_mono)
{
  const auto norms = compute_norms(ssi_mono);
  const bool converged =
      (ssi_mono.iteration_count() > 1 and
          (norms.at(SSI::L2norm::potresnorm) <= itertol_ and
              norms.at(SSI::L2norm::potincnorm) / norms.at(SSI::L2norm::potdofnorm) <= itertol_)) or
      norms.at(SSI::L2norm::potresnorm) < restol_;
  const bool exit = compute_exit(ssi_mono, converged);
  if (exit and !converged) non_converged_steps_.insert(ssi_mono.step());

  if (Core::Communication::my_mpi_rank(ssi_mono.get_comm()) == 0)
  {
    if (ssi_mono.iteration_count() == 1)
    {
      // print header
      std::cout << "Calculating initial field for electric potential\n";
      std::cout << "+------------+-------------------+--------------+--------------+\n";
      std::cout << "|- step/max -|- tol      [norm] -|-- pot-res ---|-- pot-inc ---|\n";

      // print only norm of residuals
      std::cout << "|  " << std::setw(3) << ssi_mono.iteration_count() << "/" << std::setw(3)
                << itermax_ << "   | " << std::setw(10) << std::setprecision(3) << std::scientific
                << itertol_ << "[L_2 ]  | " << std::setw(10) << std::setprecision(3)
                << std::scientific << norms.at(SSI::L2norm::potresnorm)
                << "   |      --      | (       --      , te = " << std::setw(10)
                << std::setprecision(3) << ssi_mono.dt_eval_ << ")\n";
    }

    else
    {
      std::cout << "|  " << std::setw(3) << ssi_mono.iteration_count() << "/" << std::setw(3)
                << itermax_ << "   | " << std::setw(10) << std::setprecision(3) << std::scientific
                << itertol_ << "[L_2 ]  | " << std::setw(10) << std::setprecision(3)
                << std::scientific << norms.at(SSI::L2norm::potresnorm) << "   | " << std::setw(10)
                << std::setprecision(3) << std::scientific
                << norms.at(SSI::L2norm::potincnorm) / norms.at(SSI::L2norm::potdofnorm)
                << "   | (ts = " << std::setw(10) << std::setprecision(3) << ssi_mono.dt_solve_
                << ", te = " << std::setw(10) << std::setprecision(3) << ssi_mono.dt_eval_ << ")\n";
    }

    if (exit and !converged)
    {
      std::cout << "+--------------------------------------------------------------+\n";
      std::cout << "|            >>>>>> not converged!                             |\n";
      std::cout << "+--------------------------------------------------------------+\n";
    }
    if (exit and converged)
    {
      std::cout << "+------------+-------------------+--------------+--------------+\n\n";
    }
  }

  return exit;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSI::SsiMono::ConvCheckStrategyElchScaTraManifold::get_and_check_l2_norm_scatra_manifold_conc(
    const SSI::SsiMono& ssi_mono, double& incnorm, double& resnorm, double& dofnorm) const
{
  ssi_mono.scatra_manifold()
      ->splitter()
      ->extract_other_vector(
          *ssi_mono.maps_sub_problems()->extract_vector(*ssi_mono.ssi_vectors_->increment(),
              Utils::SSIMaps::get_problem_position(Subproblem::manifold)))
      ->norm_2(&incnorm);

  ssi_mono.scatra_manifold()
      ->splitter()
      ->extract_other_vector(
          *ssi_mono.maps_sub_problems()->extract_vector(*ssi_mono.ssi_vectors_->residual(),
              Utils::SSIMaps::get_problem_position(Subproblem::manifold)))
      ->norm_2(&resnorm);

  ssi_mono.scatra_manifold()
      ->splitter()
      ->extract_other_vector(*ssi_mono.scatra_manifold()->phinp())
      ->norm_2(&dofnorm);

  check_l2_norm(incnorm, resnorm, dofnorm);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSI::SsiMono::ConvCheckStrategyElchScaTraManifold::get_and_check_l2_norm_scatra_manifold_pot(
    const SSI::SsiMono& ssi_mono, double& incnorm, double& resnorm, double& dofnorm) const
{
  ssi_mono.scatra_manifold()
      ->splitter()
      ->extract_cond_vector(
          *ssi_mono.maps_sub_problems()->extract_vector(*ssi_mono.ssi_vectors_->increment(),
              Utils::SSIMaps::get_problem_position(Subproblem::manifold)))
      ->norm_2(&incnorm);

  ssi_mono.scatra_manifold()
      ->splitter()
      ->extract_cond_vector(
          *ssi_mono.maps_sub_problems()->extract_vector(*ssi_mono.ssi_vectors_->residual(),
              Utils::SSIMaps::get_problem_position(Subproblem::manifold)))
      ->norm_2(&resnorm);

  ssi_mono.scatra_manifold()
      ->splitter()
      ->extract_cond_vector(*ssi_mono.scatra_manifold()->phinp())
      ->norm_2(&dofnorm);

  check_l2_norm(incnorm, resnorm, dofnorm);
}

/*-----------------------------------------------------------------------*
 *-----------------------------------------------------------------------*/
std::map<SSI::L2norm, double> SSI::SsiMono::ConvCheckStrategyElchScaTraManifold::compute_norms(
    const SSI::SsiMono& ssi_mono) const
{
  double concincnorm = 0.0, concresnorm = 0.0, concdofnorm = 0.0, manifoldpotdofnorm = 0.0,
         manifoldpotincnorm = 0.0, manifoldpotresnorm = 0.0, manifoldconcdofnorm = 0.0,
         manifoldconcincnorm = 0.0, manifoldconcresnorm = 0.0, potdofnorm = 0.0, potincnorm = 0.0,
         potresnorm = 0.0, structuredofnorm = 0.0, structureresnorm = 0.0, structureincnorm = 0.0;

  get_and_check_l2_norm_conc(ssi_mono, concincnorm, concresnorm, concdofnorm);
  get_and_check_l2_norm_pot(ssi_mono, potincnorm, potresnorm, potdofnorm);

  get_and_check_l2_norm_scatra_manifold_pot(
      ssi_mono, manifoldpotincnorm, manifoldpotresnorm, manifoldpotdofnorm);
  get_and_check_l2_norm_scatra_manifold_conc(
      ssi_mono, manifoldconcincnorm, manifoldconcresnorm, manifoldconcdofnorm);

  get_and_check_l2_norm_structure(ssi_mono, structureincnorm, structureresnorm, structuredofnorm);

  return {
      {SSI::L2norm::concincnorm, concincnorm},
      {SSI::L2norm::concresnorm, concresnorm},
      {SSI::L2norm::concdofnorm, concdofnorm},
      {SSI::L2norm::manifoldpotdofnorm, manifoldpotdofnorm},
      {SSI::L2norm::manifoldpotincnorm, manifoldpotincnorm},
      {SSI::L2norm::manifoldpotresnorm, manifoldpotresnorm},
      {SSI::L2norm::manifoldconcdofnorm, manifoldconcdofnorm},
      {SSI::L2norm::manifoldconcincnorm, manifoldconcincnorm},
      {SSI::L2norm::manifoldconcresnorm, manifoldconcresnorm},
      {SSI::L2norm::potdofnorm, potdofnorm},
      {SSI::L2norm::potincnorm, potincnorm},
      {SSI::L2norm::potresnorm, potresnorm},
      {SSI::L2norm::structuredofnorm, structuredofnorm},
      {SSI::L2norm::structureresnorm, structureresnorm},
      {SSI::L2norm::structureincnorm, structureincnorm},
  };
}

/*-----------------------------------------------------------------------*
 *-----------------------------------------------------------------------*/
bool SSI::SsiMono::ConvCheckStrategyElchScaTraManifold::check_convergence(
    const SSI::SsiMono& ssi_mono, const std::map<L2norm, double>& norms) const
{
  if (ssi_mono.iteration_count() > 1 and norms.at(SSI::L2norm::concresnorm) <= itertol_ and
      norms.at(SSI::L2norm::potresnorm) <= itertol_ and
      norms.at(SSI::L2norm::structureresnorm) <= itertol_ and
      norms.at(SSI::L2norm::concincnorm) / norms.at(SSI::L2norm::concdofnorm) <= itertol_ and
      norms.at(SSI::L2norm::potincnorm) / norms.at(SSI::L2norm::potdofnorm) <= itertol_ and
      norms.at(SSI::L2norm::structureincnorm) / norms.at(SSI::L2norm::structuredofnorm) <=
          itertol_ and
      norms.at(SSI::L2norm::manifoldconcresnorm) <= itertol_ and
      norms.at(SSI::L2norm::manifoldpotresnorm) <= itertol_ and
      norms.at(SSI::L2norm::manifoldpotincnorm) / norms.at(SSI::L2norm::manifoldpotdofnorm) <=
          itertol_ and
      norms.at(SSI::L2norm::manifoldconcincnorm) / norms.at(SSI::L2norm::manifoldconcdofnorm) <=
          itertol_)
    return true;

  if (norms.at(SSI::L2norm::concresnorm) < restol_ and
      norms.at(SSI::L2norm::potresnorm) < restol_ and
      norms.at(SSI::L2norm::structureresnorm) < restol_ and
      norms.at(SSI::L2norm::manifoldconcresnorm) < restol_ and
      norms.at(SSI::L2norm::manifoldpotresnorm) < restol_)
    return true;

  return false;
}

/*-----------------------------------------------------------------------*
 *-----------------------------------------------------------------------*/
void SSI::SsiMono::ConvCheckStrategyElchScaTraManifold::print_newton_iteration_information(
    const SSI::SsiMono& ssi_mono, const bool converged, const bool exit,
    const std::map<L2norm, double>& norms) const
{
  if (Core::Communication::my_mpi_rank(ssi_mono.get_comm()) != 0) return;

  if (ssi_mono.iteration_count() == 1)
  {
    // print header of convergence table to screen
    std::cout << "+------------+-------------------+--------------+--------------+--------------+"
                 "--------------+--------------+--------------+--------------+--------------+----"
                 "----------+--------------+\n";
    std::cout
        << "+------------+-------------------+--                        scatra                  "
           "       --+--        structure        --+--                        manifold          "
           "             --+\n";
    std::cout
        << "|- step/max -|- tolerance[norm] -|-- conc-res --|-- conc-inc --|-- pot-res  --|-- "
           "pot-inc  --|--   res    --|--   inc    --|-- conc-res --|-- conc-inc --|-- "
           "pot-res  --|-- pot-inc  --|\n";

    // print first line of convergence table to screen
    // solution increment not yet available during first Newton-Raphson iteration
    std::cout << "|  " << std::setw(3) << ssi_mono.iteration_count() << "/" << std::setw(3)
              << itermax_ << "   | " << std::setw(10) << std::setprecision(3) << std::scientific
              << itertol_ << "[L_2 ]  | " << std::setw(10) << std::setprecision(3)
              << std::scientific << norms.at(SSI::L2norm::concresnorm) << "   |      --      | "
              << std::setw(10) << std::setprecision(3) << std::scientific
              << norms.at(SSI::L2norm::potresnorm) << "   |      --      | " << std::setw(10)
              << std::setprecision(3) << std::scientific << norms.at(SSI::L2norm::structureresnorm)
              << "   |      --      | " << std::setw(10) << std::setprecision(3) << std::scientific
              << norms.at(SSI::L2norm::manifoldconcresnorm) << "   |      --      |  "
              << std::scientific << norms.at(SSI::L2norm::manifoldpotresnorm)
              << "   |      --      | (       --      , te = " << std::setw(10)
              << std::setprecision(3) << ssi_mono.dt_eval_ << ")\n";
  }
  else
  {
    std::cout << "|  " << std::setw(3) << ssi_mono.iteration_count() << "/" << std::setw(3)
              << itermax_ << "   | " << std::setw(10) << std::setprecision(3) << std::scientific
              << itertol_ << "[L_2 ]  | " << std::setw(10) << std::setprecision(3)
              << std::scientific << norms.at(SSI::L2norm::concresnorm) << "   | " << std::setw(10)
              << std::setprecision(3) << std::scientific
              << norms.at(SSI::L2norm::concincnorm) / norms.at(SSI::L2norm::concdofnorm) << "   | "
              << std::setw(10) << std::setprecision(3) << std::scientific
              << norms.at(SSI::L2norm::potresnorm) << "   | " << std::setw(10)
              << std::setprecision(3) << std::scientific
              << norms.at(SSI::L2norm::potincnorm) / norms.at(SSI::L2norm::potdofnorm) << "   | "
              << std::setw(10) << std::setprecision(3) << std::scientific
              << norms.at(SSI::L2norm::structureresnorm) << "   | " << std::setw(10)
              << std::setprecision(3) << std::scientific
              << norms.at(SSI::L2norm::structureincnorm) / norms.at(SSI::L2norm::structuredofnorm)
              << "   | " << std::setw(10) << std::setprecision(3) << std::scientific
              << norms.at(SSI::L2norm::manifoldconcresnorm) << "   | " << std::setw(10)
              << std::setprecision(3) << std::scientific
              << norms.at(SSI::L2norm::manifoldconcincnorm) /
                     norms.at(SSI::L2norm::manifoldconcdofnorm)
              << "   | " << std::setw(10) << std::setprecision(3) << std::scientific
              << norms.at(SSI::L2norm::manifoldpotresnorm) << "   | " << std::setw(10)
              << std::setprecision(3) << std::scientific
              << norms.at(SSI::L2norm::manifoldpotincnorm) /
                     norms.at(SSI::L2norm::manifoldpotdofnorm)
              << "   | (ts = " << std::setw(10) << std::setprecision(3) << ssi_mono.dt_solve_
              << ", te = " << std::setw(10) << std::setprecision(3) << ssi_mono.dt_eval_ << ")\n";
  }

  if (exit and !converged)
  {
    std::cout << "+------------+-------------------+--------------+--------------+--------------+"
                 "--------------+--------------+--------------+--------------+--------------+----"
                 "----------+--------------+\n";
    std::cout
        << "|              >> Newton-Raphson method has not converged after a maximum "
           "number of "
        << itermax_
        << " iterations! <<                                                                      "
           "             |\n";
    std::cout << "+------------+-------------------+--------------+--------------+--------------+--"
                 "------------+--------------+--------------+--------------+--------------+--------"
                 "------+--------------+\n";
  }

  if (exit and converged)
  {
    std::cout << "+------------+-------------------+--------------+--------------+--------------+--"
                 "------------+--------------+--------------+--------------+--------------+--------"
                 "------+--------------+\n";
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool SSI::SsiMono::ConvCheckStrategyElchScaTraManifold::exit_newton_raphson_init_pot_calc(
    const SSI::SsiMono& ssi_mono)
{
  const auto norms = compute_norms(ssi_mono);
  const bool converged =
      (ssi_mono.iteration_count() > 1 and norms.at(SSI::L2norm::potresnorm) <= itertol_ and
          norms.at(SSI::L2norm::potincnorm) / norms.at(SSI::L2norm::potdofnorm) <= itertol_ and
          norms.at(SSI::L2norm::manifoldpotresnorm) <= itertol_ and
          norms.at(SSI::L2norm::manifoldpotincnorm) / norms.at(SSI::L2norm::manifoldpotdofnorm) <=
              itertol_) or
      (norms.at(SSI::L2norm::potresnorm) < restol_ and
          norms.at(SSI::L2norm::manifoldpotresnorm) < restol_);
  const bool exit = compute_exit(ssi_mono, converged);
  if (exit and !converged) non_converged_steps_.insert(ssi_mono.step());

  if (Core::Communication::my_mpi_rank(ssi_mono.get_comm()) == 0)
  {
    if (ssi_mono.iteration_count() == 1)
    {
      // print header
      std::cout << "Calculating initial field for electric potential\n";
      std::cout
          << "+------------+-------------------+--------------+--------------+--------------+--"
             "------------+\n";
      std::cout << "+------------+-------------------+--          scatra         --|--             "
                   "manifold    --|\n";
      std::cout << "|- step/max -|- tol      [norm] -|-- pot-res  --|-- pot-inc  --|-- pot-res  "
                   "--|-- pot-inc  --|\n";

      // print only norm of residuals
      std::cout << "|  " << std::setw(3) << ssi_mono.iteration_count() << "/" << std::setw(3)
                << itermax_ << "   | " << std::setw(10) << std::setprecision(3) << std::scientific
                << itertol_ << "[L_2 ]  | " << std::setw(10) << std::setprecision(3)
                << std::scientific << norms.at(SSI::L2norm::potresnorm) << "   |      --      | "
                << std::setw(10) << std::setprecision(3) << std::scientific
                << norms.at(SSI::L2norm::manifoldpotresnorm)
                << "   |      --      | (       --      , te = " << std::setw(10)
                << std::setprecision(3) << ssi_mono.dt_eval_ << ")\n";
    }
    else
    {
      std::cout << "|  " << std::setw(3) << ssi_mono.iteration_count() << "/" << std::setw(3)
                << itermax_ << "   | " << std::setw(10) << std::setprecision(3) << std::scientific
                << itertol_ << "[L_2 ]  | " << std::setw(10) << std::setprecision(3)
                << std::scientific << norms.at(SSI::L2norm::potresnorm) << "   | " << std::setw(10)
                << std::setprecision(3) << std::scientific
                << norms.at(SSI::L2norm::potincnorm) / norms.at(SSI::L2norm::potdofnorm) << "   | "
                << std::setw(10) << std::setprecision(3) << std::scientific
                << norms.at(SSI::L2norm::manifoldpotresnorm) << "   | " << std::setw(10)
                << std::setprecision(3) << std::scientific
                << norms.at(SSI::L2norm::manifoldpotincnorm) /
                       norms.at(SSI::L2norm::manifoldpotdofnorm)
                << "   | (ts = " << std::setw(10) << std::setprecision(3) << ssi_mono.dt_solve_
                << ", te = " << std::setw(10) << std::setprecision(3) << ssi_mono.dt_eval_ << ")\n";
    }

    // warn if maximum number of iterations is reached without convergence
    if (exit and !converged)
    {
      std::cout << "+--------------------------------------------------------------------------"
                   "------------------+\n";
      std::cout << "|            >>>>>> not converged!                                           "
                   "                 |\n";
      std::cout << "+--------------------------------------------------------------------------"
                   "------------------+\n";
    }
    if (exit and converged)
    {
      std::cout << "+------------+-------------------+--------------+--------------+-----------"
                   "---+--------------+\n\n";
    }
  }

  return exit;
}
FOUR_C_NAMESPACE_CLOSE
