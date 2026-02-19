// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_structure_new_solver_factory.hpp"

#include "4C_beam3_euler_bernoulli.hpp"
#include "4C_beaminteraction_calc_utils.hpp"
#include "4C_cardiovascular0d_input.hpp"
#include "4C_contact_input.hpp"
#include "4C_fem_condition.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_discretization_nullspace.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_structure.hpp"
#include "4C_linalg_krylov_projector.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linear_solver_method.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_linear_solver_method_parameters.hpp"
#include "4C_linear_solver_method_projector.hpp"
#include "4C_utils_enum.hpp"

#include <Teuchos_ParameterList.hpp>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<Solid::SOLVER::Factory::LinSolMap> Solid::SOLVER::Factory::build_lin_solvers(
    const std::set<Inpar::Solid::ModelType>& modeltypes, const Teuchos::ParameterList& sdyn,
    Core::FE::Discretization& actdis) const
{
  // create a new standard map
  std::shared_ptr<LinSolMap> linsolvers = std::make_shared<LinSolMap>();

  std::set<Inpar::Solid::ModelType>::const_iterator mt_iter;
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
        FOUR_C_THROW("No idea which solver to use for the given model type {}", *mt_iter);
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


  // setup Krylov projection if necessary
  std::shared_ptr<Core::LinAlg::LinearSystemProjector> projector = nullptr;
  {
    // Matrix might be singular, e.g. when solid is not fully supported in a static simulation.
    // In this case, we need a basis vector for the nullspace/kernel.

    // get condition "KrylovSpaceProjection" from discretization
    std::vector<const Core::Conditions::Condition*> KSPcond;
    actdis.get_condition("KrylovSpaceProjection", KSPcond);
    int numcond = KSPcond.size();
    int numsolid = 0;

    const Core::Conditions::Condition* kspcond = nullptr;
    // check if for solid Krylov projection is required
    for (int icond = 0; icond < numcond; icond++)
    {
      const std::string& name = KSPcond[icond]->parameters().get<std::string>("DIS");
      if (name == "structure")
      {
        numsolid++;
        kspcond = KSPcond[icond];
      }
    }

    if (numsolid == 1)
    {
      const auto type = kspcond->parameters().get<std::string>("TYPE");
      if (type == "projection")
      {
        if (Core::Communication::my_mpi_rank(actdis.get_comm()) == 0)
          std::cout << "\nSetup of KrylovSpaceProjection in solid field.\n";

        const int nummodes = kspcond->parameters().get<int>("NUMMODES");

        // get rigid body mode flags for a 3-D solid: [transx transy transz rotx roty rotz]
        // Euler-Bernoulli beams have a similar ordering: [transx transy transz rot1 rot2]
        const auto* modeflags = &kspcond->parameters().get<std::vector<int>>("ONOFF");

        // get actual active mode ids given in input file
        std::vector<int> activemodeids;
        for (int rr = 0; rr < nummodes; ++rr)
        {
          if (((*modeflags)[rr]) != 0)
          {
            activemodeids.push_back(rr);
          }
        }

        const std::string* weighttype = &kspcond->parameters().get<std::string>("WEIGHTVECDEF");

        auto krylov_projector = std::make_shared<Core::LinAlg::KrylovProjector>(
            activemodeids, weighttype, actdis.dof_row_map());
        weighttype = krylov_projector->weight_type();

        if (*weighttype == "integration")
          FOUR_C_THROW("Krylov projection can currently only be done pointwise.");

        std::shared_ptr<Core::LinAlg::MultiVector<double>> c =
            krylov_projector->get_non_const_kernel();
        c->put_scalar(0.0);

        Core::LinAlg::Map nullspace_map(*actdis.dof_row_map());

        int numdof;
        int dimns;
        {
          // just grab the block information on the first element that appears
          Core::Elements::Element* dwele = actdis.l_row_element(0);
          dwele->element_type().nodal_block_information(dwele, numdof, dimns);
        }
        std::shared_ptr<Core::LinAlg::MultiVector<double>> nullspace =
            Core::FE::compute_null_space(actdis, dimns, nullspace_map);
        if (nullspace == nullptr) FOUR_C_THROW("Nullspace could not be computed successfully.");

        // sort vector of nullspace data into kernel vector c_
        std::vector<int> mode_ids = krylov_projector->modes();

        for (size_t i = 0; i < Teuchos::as<size_t>(mode_ids.size()); i++)
        {
          auto& ci = c->get_vector(i);
          auto& ni = nullspace->get_vector(mode_ids[i]);
          const size_t myLength = ci.local_length();
          for (size_t j = 0; j < myLength; j++)
          {
            ci.get_values()[j] = ni.get_values()[j];
          }
        }
        krylov_projector->fill_complete();
        projector = std::move(krylov_projector);
      }
    }
    else if (numsolid == 0)
    {
      projector = nullptr;
    }
    else
    {
      FOUR_C_THROW("Received more than one KrylovSpaceCondition for solid field");
    }

    linsolver->params().set("Projector", projector);
  }

  const auto azprectype =
      Teuchos::getIntegralValue<Core::LinearSolver::PreconditionerType>(linsolverparams, "AZPREC");

  switch (azprectype)
  {
    case Core::LinearSolver::PreconditionerType::multigrid_muelu:
    {
      compute_null_space_if_necessary(actdis, linsolver->params());
      break;
    }
    case Core::LinearSolver::PreconditionerType::block_teko:
    {
      const unsigned solid_dofset(0u);
      // Create the beam and solid maps
      std::vector<int> solid_dof_gids;
      std::vector<int> solid_node_gids;
      std::vector<int> beam_dof_gids;

      for (int i = 0; i < actdis.num_my_row_nodes(); i++)
      {
        if (const Core::Nodes::Node* node = actdis.l_row_node(i);
            BeamInteraction::Utils::is_beam_node(*node))
        {
          actdis.dof(solid_dofset, node, beam_dof_gids);
        }
        else
        {
          actdis.dof(solid_dofset, node, solid_dof_gids);
          solid_node_gids.push_back(node->id());
        }
      }

      const auto dof_row_map_solid = std::make_shared<Core::LinAlg::Map>(
          -1, solid_dof_gids.size(), solid_dof_gids.data(), 0, actdis.get_comm());
      const auto node_row_map_solid = std::make_shared<Core::LinAlg::Map>(
          -1, solid_node_gids.size(), solid_node_gids.data(), 0, actdis.get_comm());
      const auto dof_row_map_beams = std::make_shared<Core::LinAlg::Map>(
          -1, beam_dof_gids.size(), beam_dof_gids.data(), 0, actdis.get_comm());

      std::vector<std::shared_ptr<const Core::LinAlg::Map>> maps;
      maps.emplace_back(dof_row_map_solid);
      maps.emplace_back(dof_row_map_beams);

      const std::shared_ptr<Core::LinAlg::MultiMapExtractor> extractor(
          new Core::LinAlg::MultiMapExtractor(*actdis.dof_row_map(), maps));
      linsolver->params()
          .sublist("Teko Parameters")
          .set<std::shared_ptr<Core::LinAlg::MultiMapExtractor>>("extractor", extractor);

      linsolver->params()
          .sublist("Inverse1")
          .set<std::shared_ptr<Core::LinAlg::Map>>("null space: node map", node_row_map_solid);
      linsolver->params()
          .sublist("Inverse1")
          .set<std::shared_ptr<Core::LinAlg::Map>>("null space: dof map", dof_row_map_solid);
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

  const auto sol_type = Teuchos::getIntegralValue<CONTACT::SolvingStrategy>(mcparams, "STRATEGY");

  const auto sys_type = Teuchos::getIntegralValue<CONTACT::SystemType>(mcparams, "SYSTEM");

  const int lin_solver_id = mcparams.get<int>("LINEAR_SOLVER");

  return build_meshtying_contact_lin_solver(actdis, sol_type, sys_type, lin_solver_id);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Solver> Solid::SOLVER::Factory::build_meshtying_contact_lin_solver(
    Core::FE::Discretization& actdis, CONTACT::SolvingStrategy sol_type,
    CONTACT::SystemType sys_type, const int lin_solver_id)
{
  std::shared_ptr<Core::LinAlg::Solver> linsolver = nullptr;

  // get mortar information
  std::vector<const Core::Conditions::Condition*> mtcond;
  std::vector<const Core::Conditions::Condition*> ccond;
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
    case CONTACT::SystemType::saddlepoint:
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
        // if an iterative solver is chosen we need a block preconditioner
        if (prec != Core::LinearSolver::PreconditionerType::multigrid_muelu &&
            prec != Core::LinearSolver::PreconditionerType::block_teko)
          FOUR_C_THROW(
              "You have chosen an iterative linear solver. For mortar/Contact in saddlepoint "
              "formulation you have to choose a block preconditioner such as SIMPLE. Choose "
              "Teko or MueLu in the SOLVER {} block in your input file.",
              lin_solver_id);
      }

      // build meshtying/contact solver
      linsolver = std::make_shared<Core::LinAlg::Solver>(
          Global::Problem::instance()->solver_params(lin_solver_id), actdis.get_comm(),
          Global::Problem::instance()->solver_params_callback(),
          Teuchos::getIntegralValue<Core::IO::Verbositylevel>(
              Global::Problem::instance()->io_params(), "VERBOSITY"));

      compute_null_space_if_necessary(actdis, linsolver->params());

      // feed the solver object with additional information
      if (onlycontact or meshtyingandcontact)
        linsolver->params().set<bool>("CONTACT", true);
      else if (onlymeshtying)
        linsolver->params().set<bool>("MESHTYING", true);
      else
        FOUR_C_THROW(
            "this cannot be: no saddlepoint problem for beamcontact "
            "or pure structure problem.");

      if (sol_type == CONTACT::SolvingStrategy::lagmult)
      {
        // provide null space information
        if (prec == Core::LinearSolver::PreconditionerType::multigrid_muelu)
        {
          Core::LinearSolver::Parameters::compute_solver_parameters(
              actdis, linsolver->params().sublist("Inverse1").sublist("MueLu Parameters"));
          Core::LinearSolver::Parameters::compute_solver_parameters(
              actdis, linsolver->params().sublist("Inverse2").sublist("MueLu Parameters"));
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
      compute_null_space_if_necessary(actdis, linsolver->params());
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
              Global::Problem::instance()->io_params(), "VERBOSITY"),
          actdis.get_comm());
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
              Global::Problem::instance()->io_params(), "VERBOSITY"),
          actdis.get_comm());

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
          Global::Problem::instance()->io_params(), "VERBOSITY"),
      actdis.get_comm());

  // solution algorithm - direct or simple
  Cardiovascular0DInput::Cardvasc0DSolveAlgo algochoice =
      Teuchos::getIntegralValue<Cardiovascular0DInput::Cardvasc0DSolveAlgo>(
          cardvasc0dstructparams, "SOLALGORITHM");

  switch (algochoice)
  {
    case Cardiovascular0DInput::cardvasc0dsolve_direct:
      break;
    case Cardiovascular0DInput::cardvasc0dsolve_block:
    {
      linsolver->put_solver_params_to_sub_params("Inverse1",
          Global::Problem::instance()->solver_params(linsolvernumber),
          Global::Problem::instance()->solver_params_callback(),
          Teuchos::getIntegralValue<Core::IO::Verbositylevel>(
              Global::Problem::instance()->io_params(), "VERBOSITY"),
          actdis.get_comm());
      compute_null_space_if_necessary(actdis, linsolver->params().sublist("Inverse1"), true);

      linsolver->put_solver_params_to_sub_params("Inverse2",
          Global::Problem::instance()->solver_params(linsolvernumber),
          Global::Problem::instance()->solver_params_callback(),
          Teuchos::getIntegralValue<Core::IO::Verbositylevel>(
              Global::Problem::instance()->io_params(), "VERBOSITY"),
          actdis.get_comm());
      compute_null_space_if_necessary(actdis, linsolver->params().sublist("Inverse2"), true);
      break;
    }
    default:
      FOUR_C_THROW("Unknown 0D cardiovascular-structural solution technique!");
  }

  return linsolver;
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<std::map<Inpar::Solid::ModelType, std::shared_ptr<Core::LinAlg::Solver>>>
Solid::SOLVER::build_lin_solvers(const std::set<Inpar::Solid::ModelType>& modeltypes,
    const Teuchos::ParameterList& sdyn, Core::FE::Discretization& actdis)
{
  Factory factory;
  return factory.build_lin_solvers(modeltypes, sdyn, actdis);
}

FOUR_C_NAMESPACE_CLOSE
