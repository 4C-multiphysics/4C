/*-----------------------------------------------------------*/
/*! \file

\brief Factory to build the desired linear solver std::map corresponding to the active model types


\level 3

*/
/*-----------------------------------------------------------*/


#include "4C_structure_new_solver_factory.hpp"

#include "4C_beam3_euler_bernoulli.hpp"
#include "4C_beaminteraction_calc_utils.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_cardiovascular0d.hpp"
#include "4C_inpar_contact.hpp"
#include "4C_inpar_structure.hpp"
#include "4C_io_control.hpp"
#include "4C_lib_discret.hpp"
#include "4C_linalg_multiply.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linear_solver_method.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_so3_base.hpp"

#include <Teuchos_ParameterList.hpp>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
STR::SOLVER::Factory::Factory()
{
  // empty
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<STR::SOLVER::Factory::LinSolMap> STR::SOLVER::Factory::BuildLinSolvers(
    const std::set<enum INPAR::STR::ModelType>& modeltypes, const Teuchos::ParameterList& sdyn,
    DRT::Discretization& actdis) const
{
  // create a new standard map
  Teuchos::RCP<LinSolMap> linsolvers = Teuchos::rcp(new LinSolMap());

  std::set<enum INPAR::STR::ModelType>::const_iterator mt_iter;
  // loop over all model types
  for (mt_iter = modeltypes.begin(); mt_iter != modeltypes.end(); ++mt_iter)
  {
    switch (*mt_iter)
    {
      case INPAR::STR::model_structure:
      case INPAR::STR::model_springdashpot:
      case INPAR::STR::model_browniandyn:
      case INPAR::STR::model_beaminteraction:
      case INPAR::STR::model_basic_coupling:
      case INPAR::STR::model_monolithic_coupling:
      case INPAR::STR::model_partitioned_coupling:
      case INPAR::STR::model_beam_interaction_old:
      case INPAR::STR::model_constraints:
      {
        /* Check if the structural linear solver was already added and skip
         * if true. */
        LinSolMap::iterator iter = linsolvers->find(INPAR::STR::model_structure);
        if (iter == linsolvers->end())
          (*linsolvers)[INPAR::STR::model_structure] = build_structure_lin_solver(sdyn, actdis);
        break;
      }
      /* ToDo Check if this makes sense for simulations where both, meshtying and
       *      contact, are present. If we need two linsolvers, please adjust the
       *      implementation (maps for pre-conditioning, etc.). */
      case INPAR::STR::model_contact:
      case INPAR::STR::model_meshtying:
        (*linsolvers)[*mt_iter] = build_meshtying_contact_lin_solver(actdis);
        break;
      case INPAR::STR::model_lag_pen_constraint:
        (*linsolvers)[*mt_iter] = build_lag_pen_constraint_lin_solver(sdyn, actdis);
        break;
      case INPAR::STR::model_cardiovascular0d:
        (*linsolvers)[*mt_iter] = build_cardiovascular0_d_lin_solver(sdyn, actdis);
        break;
      default:
        FOUR_C_THROW("No idea which solver to use for the given model type %s",
            ModelTypeString(*mt_iter).c_str());
        break;
    }
  }

  return linsolvers;
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<CORE::LINALG::Solver> STR::SOLVER::Factory::build_structure_lin_solver(
    const Teuchos::ParameterList& sdyn, DRT::Discretization& actdis) const
{
  // get the linear solver number used for structural problems
  const int linsolvernumber = sdyn.get<int>("LINEAR_SOLVER");

  // check if the structural solver has a valid solver number
  if (linsolvernumber == (-1))
    FOUR_C_THROW(
        "no linear solver defined for structural field. "
        "Please set LINEAR_SOLVER in STRUCTURAL DYNAMIC to a valid number!");

  const Teuchos::ParameterList& linsolverparams =
      GLOBAL::Problem::Instance()->SolverParams(linsolvernumber);

  Teuchos::RCP<CORE::LINALG::Solver> linsolver =
      Teuchos::rcp(new CORE::LINALG::Solver(linsolverparams, actdis.Comm()));

  const auto azprectype =
      Teuchos::getIntegralValue<CORE::LINEAR_SOLVER::PreconditionerType>(linsolverparams, "AZPREC");

  switch (azprectype)
  {
    case CORE::LINEAR_SOLVER::PreconditionerType::multigrid_ml:
    case CORE::LINEAR_SOLVER::PreconditionerType::multigrid_ml_fluid:
    case CORE::LINEAR_SOLVER::PreconditionerType::multigrid_ml_fluid2:
    case CORE::LINEAR_SOLVER::PreconditionerType::multigrid_muelu:
    {
      actdis.compute_null_space_if_necessary(linsolver->Params());
      break;
    }
    case CORE::LINEAR_SOLVER::PreconditionerType::multigrid_muelu_beamsolid:
    {
      // Create the beam and solid maps
      std::vector<int> solidDofs(0);
      std::vector<int> beamDofs(0);

      // right now we only allow euler-bernoulli beam elements
      for (int i = 0; i < actdis.NumMyRowElements(); i++)
      {
        DRT::Element* element = actdis.lRowElement(i);

        if (BEAMINTERACTION::UTILS::IsBeamElement(*element) &&
            (element->ElementType() != DRT::ELEMENTS::Beam3ebType::Instance()))
          FOUR_C_THROW("Only beam3eb elements are currently allowed!");
      }

      for (int i = 0; i < actdis.NumMyRowNodes(); i++)
      {
        const DRT::Node* node = actdis.lRowNode(i);

        if (BEAMINTERACTION::UTILS::IsBeamNode(*node))
          actdis.Dof(node, beamDofs);
        else
          actdis.Dof(node, solidDofs);
      }

      Teuchos::RCP<Epetra_Map> rowmap1 =
          Teuchos::rcp(new Epetra_Map(-1, solidDofs.size(), solidDofs.data(), 0, actdis.Comm()));
      Teuchos::RCP<Epetra_Map> rowmap2 =
          Teuchos::rcp(new Epetra_Map(-1, beamDofs.size(), beamDofs.data(), 0, actdis.Comm()));

      linsolver->put_solver_params_to_sub_params("Inverse1", linsolverparams);
      linsolver->Params()
          .sublist("Inverse1")
          .set<Teuchos::RCP<Epetra_Map>>("null space: map", rowmap1);
      actdis.compute_null_space_if_necessary(linsolver->Params().sublist("Inverse1"));

      linsolver->put_solver_params_to_sub_params("Inverse2", linsolverparams);
      linsolver->Params()
          .sublist("Inverse2")
          .set<Teuchos::RCP<Epetra_Map>>("null space: map", rowmap2);
      actdis.compute_null_space_if_necessary(linsolver->Params().sublist("Inverse2"));

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
Teuchos::RCP<CORE::LINALG::Solver> STR::SOLVER::Factory::build_meshtying_contact_lin_solver(
    DRT::Discretization& actdis) const
{
  const Teuchos::ParameterList& mcparams = GLOBAL::Problem::Instance()->contact_dynamic_params();

  const enum INPAR::CONTACT::SolvingStrategy sol_type =
      static_cast<INPAR::CONTACT::SolvingStrategy>(
          CORE::UTILS::IntegralValue<int>(mcparams, "STRATEGY"));

  const enum INPAR::CONTACT::SystemType sys_type =
      static_cast<INPAR::CONTACT::SystemType>(CORE::UTILS::IntegralValue<int>(mcparams, "SYSTEM"));

  const int lin_solver_id = mcparams.get<int>("LINEAR_SOLVER");

  return build_meshtying_contact_lin_solver(actdis, sol_type, sys_type, lin_solver_id);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<CORE::LINALG::Solver> STR::SOLVER::Factory::build_meshtying_contact_lin_solver(
    DRT::Discretization& actdis, enum INPAR::CONTACT::SolvingStrategy sol_type,
    enum INPAR::CONTACT::SystemType sys_type, const int lin_solver_id)
{
  Teuchos::RCP<CORE::LINALG::Solver> linsolver = Teuchos::null;

  // get mortar information
  std::vector<CORE::Conditions::Condition*> mtcond(0);
  std::vector<CORE::Conditions::Condition*> ccond(0);
  actdis.GetCondition("Mortar", mtcond);
  actdis.GetCondition("Contact", ccond);
  bool onlymeshtying = false;
  bool onlycontact = false;
  bool meshtyingandcontact = false;
  if (mtcond.size() != 0 and ccond.size() != 0) meshtyingandcontact = true;
  if (mtcond.size() != 0 and ccond.size() == 0) onlymeshtying = true;
  if (mtcond.size() == 0 and ccond.size() != 0) onlycontact = true;

  // handle some special cases
  switch (sol_type)
  {
    // treat the steepest ascent strategy as a condensed system
    case INPAR::CONTACT::solution_steepest_ascent:
      sys_type = INPAR::CONTACT::system_condensed;
      break;
    // in case of the combo strategy, the actual linear solver can change during
    // the simulation and is therefore provided by the strategy
    case INPAR::CONTACT::solution_combo:
      return Teuchos::null;
    default:
      // do nothing
      break;
  }

  switch (sys_type)
  {
    case INPAR::CONTACT::system_saddlepoint:
    {
      // meshtying/contact for structure
      // check if the meshtying/contact solver has a valid solver number
      if (lin_solver_id == (-1))
        FOUR_C_THROW(
            "no linear solver defined for meshtying/contact problem. Please"
            " set LINEAR_SOLVER in CONTACT DYNAMIC to a valid number!");

      // plausibility check

      // solver can be either UMFPACK (direct solver) or an iterative solver
      const auto sol = Teuchos::getIntegralValue<CORE::LINEAR_SOLVER::SolverType>(
          GLOBAL::Problem::Instance()->SolverParams(lin_solver_id), "SOLVER");
      const auto prec = Teuchos::getIntegralValue<CORE::LINEAR_SOLVER::PreconditionerType>(
          GLOBAL::Problem::Instance()->SolverParams(lin_solver_id), "AZPREC");
      if (sol != CORE::LINEAR_SOLVER::SolverType::umfpack &&
          sol != CORE::LINEAR_SOLVER::SolverType::superlu)
      {
        // if an iterative solver is chosen we need a block preconditioner like CheapSIMPLE
        if (prec != CORE::LINEAR_SOLVER::PreconditionerType::cheap_simple &&
            prec !=
                CORE::LINEAR_SOLVER::PreconditionerType::multigrid_muelu_contactsp)  // TODO adapt
                                                                                     // error
                                                                                     // message
          FOUR_C_THROW(
              "You have chosen an iterative linear solver. For mortar/Contact in saddlepoint "
              "formulation you have to choose a block preconditioner such as SIMPLE. Choose "
              "CheapSIMPLE or MueLu_contactSP (if MueLu is available) in the SOLVER %i block in "
              "your dat file.",
              lin_solver_id);
      }

      // build meshtying/contact solver
      linsolver = Teuchos::rcp(new CORE::LINALG::Solver(
          GLOBAL::Problem::Instance()->SolverParams(lin_solver_id), actdis.Comm()));

      actdis.compute_null_space_if_necessary(linsolver->Params());

      // feed the solver object with additional information
      if (onlycontact or meshtyingandcontact)
        linsolver->Params().set<bool>("CONTACT", true);
      else if (onlymeshtying)
        linsolver->Params().set<bool>("MESHTYING", true);
      else
        FOUR_C_THROW(
            "this cannot be: no saddlepoint problem for beamcontact "
            "or pure structure problem.");

      if (sol_type == INPAR::CONTACT::solution_lagmult or
          sol_type == INPAR::CONTACT::solution_augmented or
          sol_type == INPAR::CONTACT::solution_std_lagrange or
          sol_type == INPAR::CONTACT::solution_steepest_ascent_sp)
      {
        // provide null space information
        if (prec == CORE::LINEAR_SOLVER::PreconditionerType::cheap_simple)
        {
          // Inverse2 is created within blockpreconditioners.cpp
          actdis.compute_null_space_if_necessary(
              linsolver->Params().sublist("CheapSIMPLE Parameters").sublist("Inverse1"));
        }
        else if (prec == CORE::LINEAR_SOLVER::PreconditionerType::multigrid_muelu_contactsp)
        { /* do nothing here */
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
      linsolver = Teuchos::rcp(new CORE::LINALG::Solver(
          GLOBAL::Problem::Instance()->SolverParams(lin_solver_id), actdis.Comm()));
      actdis.compute_null_space_if_necessary(linsolver->Params());
    }
    break;
  }

  return linsolver;
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<CORE::LINALG::Solver> STR::SOLVER::Factory::build_lag_pen_constraint_lin_solver(
    const Teuchos::ParameterList& sdyn, DRT::Discretization& actdis) const
{
  Teuchos::RCP<CORE::LINALG::Solver> linsolver = Teuchos::null;

  const Teuchos::ParameterList& mcparams = GLOBAL::Problem::Instance()->contact_dynamic_params();
  const Teuchos::ParameterList& strparams =
      GLOBAL::Problem::Instance()->structural_dynamic_params();

  // solution algorithm - direct, simple or Uzawa
  INPAR::STR::ConSolveAlgo algochoice =
      CORE::UTILS::IntegralValue<INPAR::STR::ConSolveAlgo>(strparams, "UZAWAALGO");

  switch (algochoice)
  {
    case INPAR::STR::consolve_direct:
    {
      const int linsolvernumber = strparams.get<int>("LINEAR_SOLVER");

      // build constraint-structural linear solver
      linsolver = Teuchos::rcp(new CORE::LINALG::Solver(
          GLOBAL::Problem::Instance()->SolverParams(linsolvernumber), actdis.Comm()));

      linsolver->Params() = CORE::LINALG::Solver::translate_solver_parameters(
          GLOBAL::Problem::Instance()->SolverParams(linsolvernumber));
    }
    break;
    case INPAR::STR::consolve_simple:
    {
      const int linsolvernumber = mcparams.get<int>("LINEAR_SOLVER");

      // build constraint-structural linear solver
      linsolver = Teuchos::rcp(new CORE::LINALG::Solver(
          GLOBAL::Problem::Instance()->SolverParams(linsolvernumber), actdis.Comm()));

      linsolver->Params() = CORE::LINALG::Solver::translate_solver_parameters(
          GLOBAL::Problem::Instance()->SolverParams(linsolvernumber));

      if (!linsolver->Params().isSublist("Belos Parameters"))
        FOUR_C_THROW("Iterative solver expected!");

      const auto prec = Teuchos::getIntegralValue<CORE::LINEAR_SOLVER::PreconditionerType>(
          GLOBAL::Problem::Instance()->SolverParams(linsolvernumber), "AZPREC");
      switch (prec)
      {
        case CORE::LINEAR_SOLVER::PreconditionerType::cheap_simple:
        {
          // add Inverse1 block for velocity dofs
          // tell Inverse1 block about nodal_block_information
          Teuchos::ParameterList& inv1 =
              linsolver->Params().sublist("CheapSIMPLE Parameters").sublist("Inverse1");
          inv1.sublist("nodal_block_information") =
              linsolver->Params().sublist("nodal_block_information");

          // calculate null space information
          actdis.compute_null_space_if_necessary(
              linsolver->Params().sublist("CheapSIMPLE Parameters").sublist("Inverse1"), true);
          actdis.compute_null_space_if_necessary(
              linsolver->Params().sublist("CheapSIMPLE Parameters").sublist("Inverse2"), true);

          linsolver->Params().sublist("CheapSIMPLE Parameters").set("Prec Type", "CheapSIMPLE");
          linsolver->Params().set("CONSTRAINT", true);
        }
        break;
        default:
          // do nothing
          break;
      }
    }
    break;
    case INPAR::STR::consolve_uzawa:
    {
      FOUR_C_THROW(
          "Uzawa-type solution techniques for constraints aren't supported anymore within the new "
          "structural time-integration!");
    }
    break;
    default:
      FOUR_C_THROW("Unknown structural-constraint solution technique!");
  }

  return linsolver;
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<CORE::LINALG::Solver> STR::SOLVER::Factory::build_cardiovascular0_d_lin_solver(
    const Teuchos::ParameterList& sdyn, DRT::Discretization& actdis) const
{
  Teuchos::RCP<CORE::LINALG::Solver> linsolver = Teuchos::null;


  const Teuchos::ParameterList& cardvasc0dstructparams =
      GLOBAL::Problem::Instance()->cardiovascular0_d_structural_params();
  const int linsolvernumber = cardvasc0dstructparams.get<int>("LINEAR_COUPLED_SOLVER");

  // build 0D cardiovascular-structural linear solver
  linsolver = Teuchos::rcp(new CORE::LINALG::Solver(
      GLOBAL::Problem::Instance()->SolverParams(linsolvernumber), actdis.Comm()));

  linsolver->Params() = CORE::LINALG::Solver::translate_solver_parameters(
      GLOBAL::Problem::Instance()->SolverParams(linsolvernumber));

  // solution algorithm - direct or simple
  INPAR::CARDIOVASCULAR0D::Cardvasc0DSolveAlgo algochoice =
      CORE::UTILS::IntegralValue<INPAR::CARDIOVASCULAR0D::Cardvasc0DSolveAlgo>(
          cardvasc0dstructparams, "SOLALGORITHM");

  switch (algochoice)
  {
    case INPAR::CARDIOVASCULAR0D::cardvasc0dsolve_direct:
      break;
    case INPAR::CARDIOVASCULAR0D::cardvasc0dsolve_simple:
    {
      const auto prec = Teuchos::getIntegralValue<CORE::LINEAR_SOLVER::PreconditionerType>(
          GLOBAL::Problem::Instance()->SolverParams(linsolvernumber), "AZPREC");
      switch (prec)
      {
        case CORE::LINEAR_SOLVER::PreconditionerType::cheap_simple:
        {
          // add Inverse1 block for velocity dofs
          // tell Inverse1 block about nodal_block_information
          Teuchos::ParameterList& inv1 =
              linsolver->Params().sublist("CheapSIMPLE Parameters").sublist("Inverse1");
          inv1.sublist("nodal_block_information") =
              linsolver->Params().sublist("nodal_block_information");

          // calculate null space information
          actdis.compute_null_space_if_necessary(
              linsolver->Params().sublist("CheapSIMPLE Parameters").sublist("Inverse1"), true);
          actdis.compute_null_space_if_necessary(
              linsolver->Params().sublist("CheapSIMPLE Parameters").sublist("Inverse2"), true);

          linsolver->Params().sublist("CheapSIMPLE Parameters").set("Prec Type", "CheapSIMPLE");
          linsolver->Params().set("CONSTRAINT", true);
        }
        break;
        default:
          // do nothing
          break;
      }
    }
    break;
    default:
      FOUR_C_THROW("Unknown 0D cardiovascular-structural solution technique!");
  }

  return linsolver;
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<std::map<enum INPAR::STR::ModelType, Teuchos::RCP<CORE::LINALG::Solver>>>
STR::SOLVER::BuildLinSolvers(const std::set<enum INPAR::STR::ModelType>& modeltypes,
    const Teuchos::ParameterList& sdyn, DRT::Discretization& actdis)
{
  Factory factory;
  return factory.BuildLinSolvers(modeltypes, sdyn, actdis);
}

FOUR_C_NAMESPACE_CLOSE