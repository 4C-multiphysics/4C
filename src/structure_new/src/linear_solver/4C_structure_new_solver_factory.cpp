// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_structure_new_solver_factory.hpp"

#include "4C_beam3_euler_bernoulli.hpp"
#include "4C_beaminteraction_calc_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_cardiovascular0d.hpp"
#include "4C_inpar_contact.hpp"
#include "4C_inpar_structure.hpp"
#include "4C_io_control.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linear_solver_method.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_linear_solver_method_parameters.hpp"

#include <Teuchos_ParameterList.hpp>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Solid::SOLVER::Factory::Factory()
{
  // empty
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<Solid::SOLVER::Factory::LinSolMap> Solid::SOLVER::Factory::build_lin_solvers(
    const std::set<enum Inpar::Solid::ModelType>& modeltypes, const Teuchos::ParameterList& sdyn,
    Core::FE::Discretization& actdis) const
{
  // create a new standard map
  std::shared_ptr<LinSolMap> linsolvers = std::make_shared<LinSolMap>();

  std::set<enum Inpar::Solid::ModelType>::const_iterator mt_iter;
  // loop over all model types
  for (mt_iter = modeltypes.begin(); mt_iter != modeltypes.end(); ++mt_iter)
  {
    switch (*mt_iter)
    {
      case Inpar::Solid::model_structure:
      case Inpar::Solid::model_springdashpot:
      case Inpar::Solid::model_browniandyn:
      case Inpar::Solid::model_beaminteraction:
      case Inpar::Solid::model_basic_coupling:
      case Inpar::Solid::model_monolithic_coupling:
      case Inpar::Solid::model_partitioned_coupling:
      case Inpar::Solid::model_beam_interaction_old:
      case Inpar::Solid::model_constraints:
      case Inpar::Solid::model_multiscale:
      {
        /* Check if the structural linear solver was already added and skip
         * if true. */
        LinSolMap::iterator iter = linsolvers->find(Inpar::Solid::model_structure);
        if (iter == linsolvers->end())
          (*linsolvers)[Inpar::Solid::model_structure] = build_structure_lin_solver(sdyn, actdis);
        break;
      }
      /* ToDo Check if this makes sense for simulations where both, meshtying and
       *      contact, are present. If we need two linsolvers, please adjust the
       *      implementation (maps for pre-conditioning, etc.). */
      case Inpar::Solid::model_contact:
      case Inpar::Solid::model_meshtying:
        (*linsolvers)[*mt_iter] = build_meshtying_contact_lin_solver(actdis);
        break;
      case Inpar::Solid::model_lag_pen_constraint:
        (*linsolvers)[*mt_iter] = build_lag_pen_constraint_lin_solver(sdyn, actdis);
        break;
      case Inpar::Solid::model_cardiovascular0d:
        (*linsolvers)[*mt_iter] = build_cardiovascular0_d_lin_solver(sdyn, actdis);
        break;
      default:
        FOUR_C_THROW("No idea which solver to use for the given model type %s",
            model_type_string(*mt_iter).c_str());
    }
  }

  return linsolvers;
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Solver> Solid::SOLVER::Factory::build_structure_lin_solver(
    const Teuchos::ParameterList& sdyn, Core::FE::Discretization& actdis) const
{
  // get the linear solver number used for structural problems
  const int linsolvernumber = sdyn.get<int>("LINEAR_SOLVER");

  // check if the structural solver has a valid solver number
  if (linsolvernumber == (-1))
    FOUR_C_THROW(
        "no linear solver defined for structural field. "
        "Please set LINEAR_SOLVER in STRUCTURAL DYNAMIC to a valid number!");

  const Teuchos::ParameterList& linsolverparams =
      Global::Problem::instance()->solver_params(linsolvernumber);

  std::shared_ptr<Core::LinAlg::Solver> linsolver = std::make_shared<Core::LinAlg::Solver>(
      linsolverparams, actdis.get_comm(), Global::Problem::instance()->solver_params_callback(),
      Teuchos::getIntegralValue<Core::IO::Verbositylevel>(
          Global::Problem::instance()->io_params(), "VERBOSITY"));

  const auto azprectype =
      Teuchos::getIntegralValue<Core::LinearSolver::PreconditionerType>(linsolverparams, "AZPREC");

  switch (azprectype)
  {
    case Core::LinearSolver::PreconditionerType::multigrid_muelu:
    {
      actdis.compute_null_space_if_necessary(linsolver->params());
      break;
    }
    case Core::LinearSolver::PreconditionerType::block_teko:
    {
      // Create the beam and solid maps
      std::vector<int> solidDofs(0);
      std::vector<int> beamDofs(0);

      for (int i = 0; i < actdis.num_my_row_nodes(); i++)
      {
        const Core::Nodes::Node* node = actdis.l_row_node(i);

        if (BeamInteraction::Utils::is_beam_node(*node))
          actdis.dof(node, beamDofs);
        else
          actdis.dof(node, solidDofs);
      }

      std::shared_ptr<Epetra_Map> rowmap1(new Epetra_Map(-1, solidDofs.size(), solidDofs.data(), 0,
          Core::Communication::as_epetra_comm(actdis.get_comm())));
      std::shared_ptr<Epetra_Map> rowmap2(new Epetra_Map(-1, beamDofs.size(), beamDofs.data(), 0,
          Core::Communication::as_epetra_comm(actdis.get_comm())));

      std::vector<std::shared_ptr<const Epetra_Map>> maps;
      maps.emplace_back(rowmap1);
      maps.emplace_back(rowmap2);

      std::shared_ptr<Core::LinAlg::MultiMapExtractor> extractor(
          new Core::LinAlg::MultiMapExtractor(*actdis.dof_row_map(), maps));
      linsolver->params()
          .sublist("Teko Parameters")
          .set<std::shared_ptr<Core::LinAlg::MultiMapExtractor>>("extractor", extractor);

      linsolver->params()
          .sublist("Inverse1")
          .set<std::shared_ptr<Epetra_Map>>("null space: map", rowmap1);
      Core::LinearSolver::Parameters::compute_solver_parameters(
          actdis, linsolver->params().sublist("Inverse1"));

      break;
    }
    default:
    {
    }
  }

  return linsolver;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Solver> Solid::SOLVER::Factory::build_meshtying_contact_lin_solver(
    Core::FE::Discretization& actdis) const
{
  const Teuchos::ParameterList& mcparams = Global::Problem::instance()->contact_dynamic_params();

  const auto sol_type =
      Teuchos::getIntegralValue<Inpar::CONTACT::SolvingStrategy>(mcparams, "STRATEGY");

  const auto sys_type = Teuchos::getIntegralValue<Inpar::CONTACT::SystemType>(mcparams, "SYSTEM");

  const int lin_solver_id = mcparams.get<int>("LINEAR_SOLVER");

  return build_meshtying_contact_lin_solver(actdis, sol_type, sys_type, lin_solver_id);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Solver> Solid::SOLVER::Factory::build_meshtying_contact_lin_solver(
    Core::FE::Discretization& actdis, enum Inpar::CONTACT::SolvingStrategy sol_type,
    enum Inpar::CONTACT::SystemType sys_type, const int lin_solver_id)
{
  std::shared_ptr<Core::LinAlg::Solver> linsolver = nullptr;

  // get mortar information
  std::vector<Core::Conditions::Condition*> mtcond(0);
  std::vector<Core::Conditions::Condition*> ccond(0);
  actdis.get_condition("Mortar", mtcond);
  actdis.get_condition("Contact", ccond);
  bool onlymeshtying = false;
  bool onlycontact = false;
  bool meshtyingandcontact = false;
  if (mtcond.size() != 0 and ccond.size() != 0) meshtyingandcontact = true;
  if (mtcond.size() != 0 and ccond.size() == 0) onlymeshtying = true;
  if (mtcond.size() == 0 and ccond.size() != 0) onlycontact = true;

  switch (sys_type)
  {
    case Inpar::CONTACT::system_saddlepoint:
    {
      // meshtying/contact for structure
      // check if the meshtying/contact solver has a valid solver number
      if (lin_solver_id == (-1))
        FOUR_C_THROW(
            "no linear solver defined for meshtying/contact problem. Please"
            " set LINEAR_SOLVER in CONTACT DYNAMIC to a valid number!");

      // plausibility check

      // solver can be either UMFPACK (direct solver) or an iterative solver
      const auto sol = Teuchos::getIntegralValue<Core::LinearSolver::SolverType>(
          Global::Problem::instance()->solver_params(lin_solver_id), "SOLVER");
      const auto prec = Teuchos::getIntegralValue<Core::LinearSolver::PreconditionerType>(
          Global::Problem::instance()->solver_params(lin_solver_id), "AZPREC");
      if (sol != Core::LinearSolver::SolverType::umfpack &&
          sol != Core::LinearSolver::SolverType::superlu)
      {
        // if an iterative solver is chosen we need a block preconditioner like CheapSIMPLE
        if (prec != Core::LinearSolver::PreconditionerType::multigrid_muelu &&
            prec != Core::LinearSolver::PreconditionerType::block_teko)
          FOUR_C_THROW(
              "You have chosen an iterative linear solver. For mortar/Contact in saddlepoint "
              "formulation you have to choose a block preconditioner such as SIMPLE. Choose "
              "CheapSIMPLE or MueLu (if MueLu is available) in the SOLVER %i block in "
              "your dat file.",
              lin_solver_id);
      }

      // build meshtying/contact solver
      linsolver = std::make_shared<Core::LinAlg::Solver>(
          Global::Problem::instance()->solver_params(lin_solver_id), actdis.get_comm(),
          Global::Problem::instance()->solver_params_callback(),
          Teuchos::getIntegralValue<Core::IO::Verbositylevel>(
              Global::Problem::instance()->io_params(), "VERBOSITY"));

      actdis.compute_null_space_if_necessary(linsolver->params());

      // feed the solver object with additional information
      if (onlycontact or meshtyingandcontact)
        linsolver->params().set<bool>("CONTACT", true);
      else if (onlymeshtying)
        linsolver->params().set<bool>("MESHTYING", true);
      else
        FOUR_C_THROW(
            "this cannot be: no saddlepoint problem for beamcontact "
            "or pure structure problem.");

      if (sol_type == Inpar::CONTACT::solution_lagmult)
      {
        // provide null space information
        if (prec == Core::LinearSolver::PreconditionerType::multigrid_muelu)
        {
          // do nothing
        }
        else if (prec == Core::LinearSolver::PreconditionerType::block_teko)
        {
          Core::LinearSolver::Parameters::compute_solver_parameters(
              actdis, linsolver->params().sublist("Inverse1"));
        }
      }
    }
    break;
    default:
    {
      // meshtying/contact for structure
      // check if the meshtying/contact solver has a valid solver number
      if (lin_solver_id == (-1))
        FOUR_C_THROW(
            "no linear solver defined for meshtying/contact problem. "
            "Please set LINEAR_SOLVER in CONTACT DYNAMIC to a valid number!");

      // build meshtying solver
      linsolver = std::make_shared<Core::LinAlg::Solver>(
          Global::Problem::instance()->solver_params(lin_solver_id), actdis.get_comm(),
          Global::Problem::instance()->solver_params_callback(),
          Teuchos::getIntegralValue<Core::IO::Verbositylevel>(
              Global::Problem::instance()->io_params(), "VERBOSITY"));
      actdis.compute_null_space_if_necessary(linsolver->params());
    }
    break;
  }

  return linsolver;
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Solver> Solid::SOLVER::Factory::build_lag_pen_constraint_lin_solver(
    const Teuchos::ParameterList& sdyn, Core::FE::Discretization& actdis) const
{
  std::shared_ptr<Core::LinAlg::Solver> linsolver = nullptr;

  const Teuchos::ParameterList& mcparams = Global::Problem::instance()->contact_dynamic_params();
  const Teuchos::ParameterList& strparams =
      Global::Problem::instance()->structural_dynamic_params();

  // solution algorithm - direct, simple or Uzawa
  auto algochoice = Teuchos::getIntegralValue<Inpar::Solid::ConSolveAlgo>(strparams, "UZAWAALGO");

  switch (algochoice)
  {
    case Inpar::Solid::consolve_direct:
    {
      const int linsolvernumber = strparams.get<int>("LINEAR_SOLVER");

      // build constraint-structural linear solver
      linsolver = std::make_shared<Core::LinAlg::Solver>(
          Global::Problem::instance()->solver_params(linsolvernumber), actdis.get_comm(),
          Global::Problem::instance()->solver_params_callback(),
          Teuchos::getIntegralValue<Core::IO::Verbositylevel>(
              Global::Problem::instance()->io_params(), "VERBOSITY"));

      linsolver->params() = Core::LinAlg::Solver::translate_solver_parameters(
          Global::Problem::instance()->solver_params(linsolvernumber),
          Global::Problem::instance()->solver_params_callback(),
          Teuchos::getIntegralValue<Core::IO::Verbositylevel>(
              Global::Problem::instance()->io_params(), "VERBOSITY"));
    }
    break;
    case Inpar::Solid::consolve_simple:
    {
      const int linsolvernumber = mcparams.get<int>("LINEAR_SOLVER");

      // build constraint-structural linear solver
      linsolver = std::make_shared<Core::LinAlg::Solver>(
          Global::Problem::instance()->solver_params(linsolvernumber), actdis.get_comm(),
          Global::Problem::instance()->solver_params_callback(),
          Teuchos::getIntegralValue<Core::IO::Verbositylevel>(
              Global::Problem::instance()->io_params(), "VERBOSITY"));

      linsolver->params() = Core::LinAlg::Solver::translate_solver_parameters(
          Global::Problem::instance()->solver_params(linsolvernumber),
          Global::Problem::instance()->solver_params_callback(),
          Teuchos::getIntegralValue<Core::IO::Verbositylevel>(
              Global::Problem::instance()->io_params(), "VERBOSITY"));

      if (!linsolver->params().isSublist("Belos Parameters"))
        FOUR_C_THROW("Iterative solver expected!");

      const auto prec = Teuchos::getIntegralValue<Core::LinearSolver::PreconditionerType>(
          Global::Problem::instance()->solver_params(linsolvernumber), "AZPREC");
      switch (prec)
      {
        case Core::LinearSolver::PreconditionerType::block_teko:
        {
          Core::LinearSolver::Parameters::compute_solver_parameters(
              actdis, linsolver->params().sublist("Inverse1"));
          break;
        }
        default:
          // do nothing
          break;
      }
    }
    break;
    case Inpar::Solid::consolve_uzawa:
    {
      FOUR_C_THROW(
          "Uzawa-type solution techniques for constraints aren't supported anymore within the new "
          "structural time-integration!");
    }
    default:
      FOUR_C_THROW("Unknown structural-constraint solution technique!");
  }

  return linsolver;
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Solver> Solid::SOLVER::Factory::build_cardiovascular0_d_lin_solver(
    const Teuchos::ParameterList& sdyn, Core::FE::Discretization& actdis) const
{
  std::shared_ptr<Core::LinAlg::Solver> linsolver = nullptr;


  const Teuchos::ParameterList& cardvasc0dstructparams =
      Global::Problem::instance()->cardiovascular0_d_structural_params();
  const int linsolvernumber = cardvasc0dstructparams.get<int>("LINEAR_COUPLED_SOLVER");

  // build 0D cardiovascular-structural linear solver
  linsolver = std::make_shared<Core::LinAlg::Solver>(
      Global::Problem::instance()->solver_params(linsolvernumber), actdis.get_comm(),
      Global::Problem::instance()->solver_params_callback(),
      Teuchos::getIntegralValue<Core::IO::Verbositylevel>(
          Global::Problem::instance()->io_params(), "VERBOSITY"));

  linsolver->params() = Core::LinAlg::Solver::translate_solver_parameters(
      Global::Problem::instance()->solver_params(linsolvernumber),
      Global::Problem::instance()->solver_params_callback(),
      Teuchos::getIntegralValue<Core::IO::Verbositylevel>(
          Global::Problem::instance()->io_params(), "VERBOSITY"));

  // solution algorithm - direct or simple
  Inpar::Cardiovascular0D::Cardvasc0DSolveAlgo algochoice =
      Teuchos::getIntegralValue<Inpar::Cardiovascular0D::Cardvasc0DSolveAlgo>(
          cardvasc0dstructparams, "SOLALGORITHM");

  switch (algochoice)
  {
    case Inpar::Cardiovascular0D::cardvasc0dsolve_direct:
      break;
    case Inpar::Cardiovascular0D::cardvasc0dsolve_block:
    {
      linsolver->put_solver_params_to_sub_params("Inverse1",
          Global::Problem::instance()->solver_params(linsolvernumber),
          Global::Problem::instance()->solver_params_callback(),
          Teuchos::getIntegralValue<Core::IO::Verbositylevel>(
              Global::Problem::instance()->io_params(), "VERBOSITY"));
      actdis.compute_null_space_if_necessary(linsolver->params().sublist("Inverse1"), true);

      linsolver->put_solver_params_to_sub_params("Inverse2",
          Global::Problem::instance()->solver_params(linsolvernumber),
          Global::Problem::instance()->solver_params_callback(),
          Teuchos::getIntegralValue<Core::IO::Verbositylevel>(
              Global::Problem::instance()->io_params(), "VERBOSITY"));
      actdis.compute_null_space_if_necessary(linsolver->params().sublist("Inverse2"), true);
      break;
    }
    default:
      FOUR_C_THROW("Unknown 0D cardiovascular-structural solution technique!");
  }

  return linsolver;
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<std::map<enum Inpar::Solid::ModelType, std::shared_ptr<Core::LinAlg::Solver>>>
Solid::SOLVER::build_lin_solvers(const std::set<enum Inpar::Solid::ModelType>& modeltypes,
    const Teuchos::ParameterList& sdyn, Core::FE::Discretization& actdis)
{
  Factory factory;
  return factory.build_lin_solvers(modeltypes, sdyn, actdis);
}

FOUR_C_NAMESPACE_CLOSE
