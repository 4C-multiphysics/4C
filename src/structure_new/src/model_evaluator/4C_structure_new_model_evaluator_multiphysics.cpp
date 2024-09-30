/*-----------------------------------------------------------*/
/*! \file

\brief Base class for modelevaluators in partitioned algorithms.


\level 3

*/
/*-----------------------------------------------------------*/


#include "4C_structure_new_model_evaluator_multiphysics.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Solid::ModelEvaluator::Multiphysics::Multiphysics() : active_mt_(mt_none)
{
  // empty constructor
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::Multiphysics::init(
    const Teuchos::RCP<Solid::ModelEvaluator::Data>& eval_data_ptr,
    const Teuchos::RCP<Solid::TimeInt::BaseDataGlobalState>& gstate_ptr,
    const Teuchos::RCP<Solid::TimeInt::BaseDataIO>& gio_ptr,
    const Teuchos::RCP<Solid::Integrator>& int_ptr,
    const Teuchos::RCP<const Solid::TimeInt::Base>& timint_ptr, const int& dof_offset)
{
  Solid::ModelEvaluator::Generic::init(
      eval_data_ptr, gstate_ptr, gio_ptr, int_ptr, timint_ptr, dof_offset);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::Multiphysics::setup()
{
  check_init();
  issetup_ = true;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Solid::ModelEvaluator::Multiphysics::reset(const Core::LinAlg::Vector& x)
{
  check_active_model_type();

  get_model_evaluator_from_map(active_mt_)->reset(x);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool Solid::ModelEvaluator::Multiphysics::assemble_force(
    Core::LinAlg::Vector& f, const double& timefac_np) const
{
  check_active_model_type();

  get_model_evaluator_from_map(active_mt_)->assemble_force(f, timefac_np);
  return true;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool Solid::ModelEvaluator::Multiphysics::assemble_jacobian(
    Core::LinAlg::SparseOperator& jac, const double& timefac_np) const
{
  check_active_model_type();

  get_model_evaluator_from_map(active_mt_)->assemble_jacobian(jac, timefac_np);
  return true;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool Solid::ModelEvaluator::Multiphysics::evaluate_force()
{
  check_active_model_type();

  get_model_evaluator_from_map(active_mt_)->evaluate_force();
  return true;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool Solid::ModelEvaluator::Multiphysics::evaluate_stiff()
{
  check_active_model_type();

  get_model_evaluator_from_map(active_mt_)->evaluate_stiff();
  return true;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool Solid::ModelEvaluator::Multiphysics::evaluate_force_stiff()
{
  check_active_model_type();

  get_model_evaluator_from_map(active_mt_)->evaluate_force_stiff();
  return true;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Solid::ModelEvaluator::Multiphysics::update_step_state(const double& timefac_n)
{
  check_active_model_type();

  get_model_evaluator_from_map(active_mt_)->update_step_state(timefac_n);
}

FOUR_C_NAMESPACE_CLOSE
