// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_structure_new_utils.hpp"

#include "4C_constraint_lagpenconstraint_noxinterface.hpp"
#include "4C_constraint_manager.hpp"
#include "4C_contact_abstract_strategy.hpp"
#include "4C_contact_meshtying_abstract_strategy.hpp"
#include "4C_contact_meshtying_noxinterface.hpp"
#include "4C_contact_noxinterface.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_structure.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_solver_nonlin_nox_aux.hpp"
#include "4C_solver_nonlin_nox_constraint_interface_preconditioner.hpp"
#include "4C_solver_nonlin_nox_constraint_interface_required.hpp"
#include "4C_structure_new_impl_genalpha.hpp"
#include "4C_structure_new_integrator.hpp"
#include "4C_structure_new_model_evaluator_contact.hpp"
#include "4C_structure_new_model_evaluator_lagpenconstraint.hpp"
#include "4C_structure_new_model_evaluator_meshtying.hpp"
#include "4C_structure_new_nln_linearsystem_scaling.hpp"
#include "4C_structure_new_timint_basedatasdyn.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
NOX::Nln::LinSystem::ConditionNumber Solid::Nln::convert2_nox_condition_number_type(
    const Inpar::Solid::ConditionNumber stype)
{
  switch (stype)
  {
    case Inpar::Solid::ConditionNumber::max_min_ev_ratio:
      return NOX::Nln::LinSystem::ConditionNumber::max_min_ev_ratio;
    case Inpar::Solid::ConditionNumber::one_norm:
      return NOX::Nln::LinSystem::ConditionNumber::one_norm;
    case Inpar::Solid::ConditionNumber::inf_norm:
      return NOX::Nln::LinSystem::ConditionNumber::inf_norm;
    default:
      FOUR_C_THROW("No known conversion.");
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
enum ::NOX::Abstract::Vector::NormType Solid::Nln::convert2_nox_norm_type(
    const Inpar::Solid::VectorNorm& normtype)
{
  enum ::NOX::Abstract::Vector::NormType nox_normtype = ::NOX::Abstract::Vector::TwoNorm;

  switch (normtype)
  {
    case Inpar::Solid::norm_l2:
      nox_normtype = ::NOX::Abstract::Vector::TwoNorm;
      break;
    case Inpar::Solid::norm_l1:
      nox_normtype = ::NOX::Abstract::Vector::OneNorm;
      break;
    case Inpar::Solid::norm_inf:
      nox_normtype = ::NOX::Abstract::Vector::MaxNorm;
      break;
    case Inpar::Solid::norm_rms:
    case Inpar::Solid::norm_vague:
    default:
      FOUR_C_THROW("Unknown conversion for the given vector norm type: \" {} \"!", normtype);
      break;
  }  // switch case normtype

  return nox_normtype;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::Nln::convert_model_type2_sol_type(std::vector<NOX::Nln::SolutionType>& soltypes,
    std::map<NOX::Nln::SolutionType, Teuchos::RCP<Core::LinAlg::Solver>>& slinsolvers,
    const std::set<Inpar::Solid::ModelType>& modeltypes,
    const std::map<Inpar::Solid::ModelType, std::shared_ptr<Core::LinAlg::Solver>>& mlinsolvers)
{
  // initialize the vector and/or force the length to zero
  if (soltypes.size() > 0)
  {
    soltypes.clear();
    slinsolvers.clear();
  }

  // pre-set the vector size
  soltypes.reserve(modeltypes.size());

  // The strings of the different enums have to fit!
  std::set<Inpar::Solid::ModelType>::const_iterator mt_iter;
  for (mt_iter = modeltypes.begin(); mt_iter != modeltypes.end(); ++mt_iter)
  {
    const NOX::Nln::SolutionType soltype = convert_model_type2_sol_type(*mt_iter);

    soltypes.push_back(soltype);
    // copy the linsolver pointers into the new map
    if (mlinsolvers.find(*mt_iter) != mlinsolvers.end())
      slinsolvers[soltype] = Teuchos::rcpFromRef(*mlinsolvers.at(*mt_iter));
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
NOX::Nln::SolutionType Solid::Nln::convert_model_type2_sol_type(
    const Inpar::Solid::ModelType& modeltype, const bool& do_check)
{
  NOX::Nln::SolutionType soltype = NOX::Nln::sol_unknown;
  switch (modeltype)
  {
    case Inpar::Solid::model_structure:
    case Inpar::Solid::model_springdashpot:
    case Inpar::Solid::model_basic_coupling:
    case Inpar::Solid::model_monolithic_coupling:
    case Inpar::Solid::model_partitioned_coupling:
    case Inpar::Solid::model_beam_interaction_old:
    case Inpar::Solid::model_browniandyn:
    case Inpar::Solid::model_beaminteraction:
    case Inpar::Solid::model_constraints:
    case Inpar::Solid::model_multiscale:
      soltype = NOX::Nln::sol_structure;
      break;
    case Inpar::Solid::model_contact:
      soltype = NOX::Nln::sol_contact;
      break;
    case Inpar::Solid::model_meshtying:
      soltype = NOX::Nln::sol_meshtying;
      break;
    case Inpar::Solid::model_cardiovascular0d:
      soltype = NOX::Nln::sol_cardiovascular0d;
      break;
    case Inpar::Solid::model_lag_pen_constraint:
      soltype = NOX::Nln::sol_lag_pen_constraint;
      break;
    default:
      // check if the corresponding enum could be found.
      if (do_check)
        FOUR_C_THROW(
            "The corresponding solution-type was not found. "
            "Given string: {}",
            modeltype);
      break;
  }

  return soltype;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Inpar::Solid::ModelType Solid::Nln::convert_sol_type2_model_type(
    const NOX::Nln::SolutionType& soltype, const bool& do_check)
{
  Inpar::Solid::ModelType modeltype = Inpar::Solid::model_vague;
  switch (soltype)
  {
    case NOX::Nln::sol_structure:
      modeltype = Inpar::Solid::model_structure;
      break;
    case NOX::Nln::sol_contact:
      modeltype = Inpar::Solid::model_contact;
      break;
    case NOX::Nln::sol_meshtying:
      modeltype = Inpar::Solid::model_meshtying;
      break;
    case NOX::Nln::sol_cardiovascular0d:
      modeltype = Inpar::Solid::model_cardiovascular0d;
      break;
    case NOX::Nln::sol_lag_pen_constraint:
      modeltype = Inpar::Solid::model_lag_pen_constraint;
      break;
    case NOX::Nln::sol_beaminteraction_lm:
      modeltype = Inpar::Solid::model_beaminteraction;
      break;
    default:
      // check if the corresponding enum could be found.
      if (do_check)
        FOUR_C_THROW(
            "The corresponding model-type was not found. "
            "Given string: {}",
            NOX::Nln::solution_type_to_string(soltype).c_str());
      break;
  }

  return modeltype;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Inpar::Solid::ModelType Solid::Nln::convert_quantity_type2_model_type(
    const NOX::Nln::StatusTest::QuantityType& qtype, const bool& do_check)
{
  const NOX::Nln::SolutionType st = NOX::Nln::Aux::convert_quantity_type_to_solution_type(qtype);
  return convert_sol_type2_model_type(st, do_check);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Inpar::Solid::EleTech Solid::Nln::convert_quantity_type2_ele_tech(
    const NOX::Nln::StatusTest::QuantityType& qtype)
{
  Inpar::Solid::EleTech eletech;
  switch (qtype)
  {
    case NOX::Nln::StatusTest::quantity_eas:
      eletech = Inpar::Solid::EleTech::eas;
      break;
    default:
      FOUR_C_THROW("Cannot convert QuantityType {} to EleTech.",
          NOX::Nln::StatusTest::quantity_type_to_string(qtype));
      break;
  }

  return eletech;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
NOX::Nln::OptimizationProblemType Solid::Nln::optimization_type(
    const std::vector<NOX::Nln::SolutionType>& soltypes)
{
  NOX::Nln::OptimizationProblemType opttype = NOX::Nln::opt_unconstrained;
  std::vector<NOX::Nln::SolutionType>::const_iterator st_iter;

  for (st_iter = soltypes.begin(); st_iter != soltypes.end(); ++st_iter)
  {
    switch (*st_iter)
    {
      // -----------------------------------
      // Inequality constraint
      // active set decision necessary
      // saddle point structure or condensed
      // -----------------------------------
      case NOX::Nln::sol_contact:
        return NOX::Nln::opt_inequality_constrained;
        break;
      // -----------------------------------
      // Equality constraint
      // no active set decision necessary
      // saddle point structure or condensed
      // -----------------------------------
      case NOX::Nln::sol_meshtying:
      case NOX::Nln::sol_lag_pen_constraint:
        opttype = NOX::Nln::opt_equality_constrained;
        break;
      // -----------------------------------
      // Unconstrained problem
      // pure structural problem
      // no saddle point structure
      // -----------------------------------
      case NOX::Nln::sol_structure:
      case NOX::Nln::sol_cardiovascular0d:
      default:
        break;
    }
  }

  return opttype;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::Nln::create_constraint_interfaces(NOX::Nln::CONSTRAINT::ReqInterfaceMap& iconstr,
    Solid::Integrator& integrator, const std::vector<NOX::Nln::SolutionType>& soltypes)
{
  if (iconstr.size() > 0) iconstr.clear();

  std::vector<NOX::Nln::SolutionType>::const_iterator st_iter;
  for (st_iter = soltypes.begin(); st_iter != soltypes.end(); ++st_iter)
  {
    switch (*st_iter)
    {
      case NOX::Nln::sol_contact:
      {
        Solid::ModelEvaluator::Generic& model = integrator.evaluator(Inpar::Solid::model_contact);
        Solid::ModelEvaluator::Contact& contact_model =
            dynamic_cast<Solid::ModelEvaluator::Contact&>(model);
        iconstr[NOX::Nln::sol_contact] =
            Teuchos::rcpFromRef(*contact_model.strategy_ptr()->nox_interface_ptr());
        break;
      }
      case NOX::Nln::sol_meshtying:
      {
        Solid::ModelEvaluator::Generic& model = integrator.evaluator(Inpar::Solid::model_meshtying);
        Solid::ModelEvaluator::Meshtying& mt_model =
            dynamic_cast<Solid::ModelEvaluator::Meshtying&>(model);
        iconstr[NOX::Nln::sol_meshtying] =
            Teuchos::rcpFromRef(*mt_model.strategy_ptr()->nox_interface_ptr());
        break;
      }
      case NOX::Nln::sol_lag_pen_constraint:
      {
        Solid::ModelEvaluator::Generic& model =
            integrator.evaluator(Inpar::Solid::model_lag_pen_constraint);
        Solid::ModelEvaluator::LagPenConstraint& lagpenconstraint_model =
            dynamic_cast<Solid::ModelEvaluator::LagPenConstraint&>(model);
        iconstr[NOX::Nln::sol_lag_pen_constraint] =
            Teuchos::rcpFromRef(*lagpenconstraint_model.nox_interface_ptr());
        break;
      }
      default:
        break;
    }
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::Nln::create_constraint_preconditioner(
    NOX::Nln::CONSTRAINT::PrecInterfaceMap& iconstr_prec, Solid::Integrator& integrator,
    const std::vector<NOX::Nln::SolutionType>& soltypes)
{
  if (iconstr_prec.size() > 0) iconstr_prec.clear();

  std::vector<NOX::Nln::SolutionType>::const_iterator st_iter;
  for (st_iter = soltypes.begin(); st_iter != soltypes.end(); ++st_iter)
  {
    switch (*st_iter)
    {
      case NOX::Nln::sol_contact:
      {
        Solid::ModelEvaluator::Generic& model = integrator.evaluator(Inpar::Solid::model_contact);
        Solid::ModelEvaluator::Contact& contact_model =
            dynamic_cast<Solid::ModelEvaluator::Contact&>(model);
        /* Actually we use the underlying Mortar::StrategyBase as Preconditioner
         * interface. Nevertheless, the implementations can differ for the
         * contact/meshtying cases. */
        iconstr_prec[NOX::Nln::sol_contact] = Teuchos::rcpFromRef(*contact_model.strategy_ptr());
        break;
      }
      case NOX::Nln::sol_meshtying:
      {
        Solid::ModelEvaluator::Generic& model = integrator.evaluator(Inpar::Solid::model_meshtying);
        Solid::ModelEvaluator::Meshtying& mt_model =
            dynamic_cast<Solid::ModelEvaluator::Meshtying&>(model);
        iconstr_prec[NOX::Nln::sol_meshtying] = Teuchos::rcpFromRef(*mt_model.strategy_ptr());
        break;
      }
      case NOX::Nln::sol_lag_pen_constraint:
      {
        Solid::ModelEvaluator::Generic& model =
            integrator.evaluator(Inpar::Solid::model_lag_pen_constraint);
        Solid::ModelEvaluator::LagPenConstraint& lagpenconstraint_model =
            dynamic_cast<Solid::ModelEvaluator::LagPenConstraint&>(model);
        iconstr_prec[NOX::Nln::sol_lag_pen_constraint] =
            Teuchos::rcpFromRef(*lagpenconstraint_model.nox_interface_prec_ptr());
        break;
      }
      default:
        // do nothing
        break;
    }  // switch (*st_iter)
  }
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::Nln::create_scaling(std::shared_ptr<NOX::Nln::Scaling>& iscale,
    const Solid::TimeInt::BaseDataSDyn& DataSDyn, Solid::TimeInt::BaseDataGlobalState& GState)
{
  if (DataSDyn.get_stc_algo_type() != Inpar::Solid::stc_inactive)
    iscale = std::make_shared<Solid::Nln::LinSystem::StcScaling>(DataSDyn, GState);
}

void Solid::compute_generalized_alpha_parameters(Solid::IMPLICIT::GenAlpha::Coefficients& coeffs)
{
  // ------ check if the user provide RHO_INF and any other parameters at the same time
  if (((coeffs.beta_ != -1.0) or (coeffs.gamma_ != -1.0) or (coeffs.alpham_ != -1.0) or
          (coeffs.alphaf_ != -1.0)) and
      (coeffs.rhoinf_ != -1.0))
    FOUR_C_THROW(
        "There are two ways to provide GenAlpha parameters:\n"
        "- You can choose to only provide RHO_INF as the spectral radius."
        "In this way, no other parameters are allowed.\n"
        "- You may also specify all the four parameters"
        "In this way, you MUST set RHO_INF as -1.0");

  // ------ rho_inf set to -1.0--> use the four parameters provided by the user -----------------
  else if (coeffs.rhoinf_ == -1.0)
  {
    if ((coeffs.alpham_ < 0.0) or (coeffs.alpham_ >= 1.0))
      FOUR_C_THROW("alpham out of range [0.0,1.0)");
    if ((coeffs.alphaf_ < 0.0) or (coeffs.alphaf_ >= 1.0))
      FOUR_C_THROW("alphaf out of range [0.0,1.0)");
    if ((coeffs.beta_ <= 0.0) or (coeffs.beta_ > 0.5)) FOUR_C_THROW("beta out of range (0.0,0.5]");
    if ((coeffs.gamma_ <= 0.0) or (coeffs.gamma_ > 1.0))
      FOUR_C_THROW("gamma out of range (0.0,1.0]");
  }

  // ------ rho_inf out of [0,1]--> report error
  else if ((coeffs.rhoinf_ < 0.0) or (coeffs.rhoinf_ > 1.0))
    FOUR_C_THROW("rho_inf out of range [0.0,1.0]");

  // ------ rho_inf specified --> calculate optimal parameters -----------------
  else
  {
    coeffs.alpham_ = (2.0 * coeffs.rhoinf_ - 1.0) / (coeffs.rhoinf_ + 1.0);
    coeffs.alphaf_ = coeffs.rhoinf_ / (coeffs.rhoinf_ + 1.0);
    coeffs.beta_ =
        0.25 * (1.0 - coeffs.alpham_ + coeffs.alphaf_) * (1.0 - coeffs.alpham_ + coeffs.alphaf_);
    coeffs.gamma_ = 0.5 - coeffs.alpham_ + coeffs.alphaf_;
  };
}

FOUR_C_NAMESPACE_CLOSE
