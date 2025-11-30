// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_structure_new_model_evaluator_factory.hpp"

#include "4C_beaminteraction_str_model_evaluator.hpp"
#include "4C_browniandyn_str_model_evaluator.hpp"
#include "4C_cardiovascular0d_structure_new_model_evaluator.hpp"
#include "4C_constraint_framework_model_evaluator.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_structure.hpp"
#include "4C_structure_new_model_evaluator_contact.hpp"
#include "4C_structure_new_model_evaluator_lagpenconstraint.hpp"
#include "4C_structure_new_model_evaluator_meshtying.hpp"
#include "4C_structure_new_model_evaluator_multiscale.hpp"
#include "4C_structure_new_model_evaluator_springdashpot.hpp"
#include "4C_structure_new_model_evaluator_structure.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Solid::ModelEvaluator::Factory::Factory()
{
  // empty constructor
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<Solid::ModelEvaluatorManager::Map>
Solid::ModelEvaluator::Factory::build_model_evaluators(
    const std::set<Inpar::Solid::ModelType>& modeltypes,
    const std::shared_ptr<Solid::ModelEvaluator::Generic>& coupling_model_ptr) const
{
  // create a new standard map
  std::shared_ptr<Solid::ModelEvaluatorManager::Map> model_map =
      std::make_shared<Solid::ModelEvaluatorManager::Map>();

  std::set<Inpar::Solid::ModelType>::const_iterator mt_iter;
  for (mt_iter = modeltypes.begin(); mt_iter != modeltypes.end(); ++mt_iter)
  {
    switch (*mt_iter)
    {
      case Inpar::Solid::model_structure:
        (*model_map)[*mt_iter] = build_structure_model_evaluator();
        break;
      case Inpar::Solid::model_springdashpot:
        (*model_map)[*mt_iter] = std::make_shared<Solid::ModelEvaluator::SpringDashpot>();
        break;
      case Inpar::Solid::model_browniandyn:
        (*model_map)[*mt_iter] = std::make_shared<Solid::ModelEvaluator::BrownianDyn>();
        break;
      case Inpar::Solid::model_beaminteraction:
        (*model_map)[*mt_iter] = std::make_shared<Solid::ModelEvaluator::BeamInteraction>();
        break;
      case Inpar::Solid::model_contact:
      {
        (*model_map)[*mt_iter] = build_contact_model_evaluator();
        break;
      }
      case Inpar::Solid::model_lag_pen_constraint:
        (*model_map)[*mt_iter] = std::make_shared<Solid::ModelEvaluator::LagPenConstraint>();
        break;
      case Inpar::Solid::model_cardiovascular0d:
        (*model_map)[*mt_iter] = std::make_shared<Solid::ModelEvaluator::Cardiovascular0D>();
        break;
      case Inpar::Solid::model_monolithic_coupling:
      {
        if (!coupling_model_ptr)
          FOUR_C_THROW("The monolithic coupling model evaluator is not defined.");
        (*model_map)[*mt_iter] = coupling_model_ptr;
        break;
      }
      case Inpar::Solid::model_partitioned_coupling:
      {
        if (!coupling_model_ptr)
          FOUR_C_THROW("The partitioned coupling model evaluator is not defined.");
        (*model_map)[*mt_iter] = coupling_model_ptr;
        break;
      }
      case Inpar::Solid::model_basic_coupling:
      {
        if (!coupling_model_ptr) FOUR_C_THROW("The basic coupling model evaluator is not defined.");
        (*model_map)[*mt_iter] = coupling_model_ptr;
        break;
      }
      case Inpar::Solid::model_meshtying:
        (*model_map)[*mt_iter] = std::make_shared<Solid::ModelEvaluator::Meshtying>();
        break;
      case Inpar::Solid::model_constraints:
        (*model_map)[*mt_iter] = std::make_shared<Solid::ModelEvaluator::Constraint>();
        break;
      case Inpar::Solid::model_multiscale:
        (*model_map)[*mt_iter] = std::make_shared<Solid::ModelEvaluator::Multiscale>();
        break;
      default:
        FOUR_C_THROW("Not yet implemented!");
        break;
    }
  }

  return model_map;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<Solid::ModelEvaluator::Generic>
Solid::ModelEvaluator::Factory::build_contact_model_evaluator() const
{
  return std::make_shared<Solid::ModelEvaluator::Contact>();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<Solid::ModelEvaluator::Generic>
Solid::ModelEvaluator::Factory::build_structure_model_evaluator() const
{
  return std::make_shared<Solid::ModelEvaluator::Structure>();
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<Solid::ModelEvaluatorManager::Map> Solid::ModelEvaluator::build_model_evaluators(
    const std::set<Inpar::Solid::ModelType>& modeltypes,
    const std::shared_ptr<Solid::ModelEvaluator::Generic>& coupling_model_ptr)
{
  Factory factory;
  return factory.build_model_evaluators(modeltypes, coupling_model_ptr);
}

FOUR_C_NAMESPACE_CLOSE
