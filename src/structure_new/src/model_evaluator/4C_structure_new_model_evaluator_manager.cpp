// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_structure_new_model_evaluator_manager.hpp"

#include "4C_linalg_blocksparsematrix.hpp"  // debugging
#include "4C_linalg_sparseoperator.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_structure_new_integrator.hpp"
#include "4C_structure_new_model_evaluator_data.hpp"
#include "4C_structure_new_model_evaluator_factory.hpp"
#include "4C_structure_new_model_evaluator_structure.hpp"
#include "4C_structure_new_timint_base.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Solid::ModelEvaluatorManager::ModelEvaluatorManager()
    : isinit_(false),
      issetup_(false),
      me_map_ptr_(nullptr),
      me_vec_ptr_(nullptr),
      eval_data_ptr_(nullptr),
      sdyn_ptr_(nullptr),
      gstate_ptr_(nullptr),
      gio_ptr_(nullptr),
      int_ptr_(nullptr),
      timint_ptr_(nullptr)
{
  // empty constructor
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluatorManager::check_init_setup() const
{
  FOUR_C_ASSERT(is_init() and is_setup(), "Call init() and setup() first!");
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluatorManager::check_init() const
{
  FOUR_C_ASSERT(is_init(), "Call init() first!");
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluatorManager::init(
    const std::shared_ptr<Solid::ModelEvaluator::Data>& eval_data_ptr,
    const std::shared_ptr<Solid::TimeInt::BaseDataSDyn>& sdyn_ptr,
    const std::shared_ptr<Solid::TimeInt::BaseDataGlobalState>& gstate_ptr,
    const std::shared_ptr<Solid::TimeInt::BaseDataIO>& gio_ptr,
    const std::shared_ptr<Solid::Integrator>& int_ptr,
    const std::shared_ptr<const Solid::TimeInt::Base>& timint_ptr)
{
  issetup_ = false;

  eval_data_ptr_ = eval_data_ptr;
  sdyn_ptr_ = sdyn_ptr;
  gstate_ptr_ = gstate_ptr;
  gio_ptr_ = gio_ptr;
  int_ptr_ = int_ptr;
  timint_ptr_ = timint_ptr;

  isinit_ = true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluatorManager::setup()
{
  check_init();

  me_map_ptr_ = Solid::ModelEvaluator::build_model_evaluators(
      sdyn_ptr_->get_model_types(), sdyn_ptr_->coupling_model_ptr());

  me_vec_ptr_ = transform_to_vector(*me_map_ptr_);

  int dof_offset = 0;
  for (const auto& [model_type, model_evaluator] : *me_map_ptr_)
  {
    model_evaluator->init(eval_data_ptr_, gstate_ptr_, gio_ptr_, int_ptr_, timint_ptr_, dof_offset);
    model_evaluator->setup();
    // setup the block information for saddle point problems
    dof_offset = gstate_ptr_->setup_block_information(*model_evaluator, model_type);
  }
  gstate_ptr_->setup_multi_map_extractor();
  gstate_ptr_->setup_element_technology_map_extractors();

  issetup_ = true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluatorManager::setup_multi_map_extractor()
{
  // setup the block information for saddle point problems
  for (const auto& me_iter : *me_vec_ptr_)
    gstate_ptr_->setup_block_information(*me_iter, me_iter->type());

  gstate_ptr_->setup_multi_map_extractor();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Solid::ModelEvaluatorManager::initialize_inertia_and_damping(
    const Core::LinAlg::Vector<double>& x, Core::LinAlg::SparseOperator& jac)
{
  check_init_setup();

  // initialize stiffness matrix to zero
  jac.zero();
  // get structural model evaluator
  Solid::ModelEvaluator::Structure& str_model =
      dynamic_cast<Solid::ModelEvaluator::Structure&>(evaluator(Inpar::Solid::model_structure));

  str_model.reset(x);

  return str_model.initialize_inertia_and_damping();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Solid::ModelEvaluatorManager::assemble_force(const double timefac_np,
    Core::LinAlg::Vector<double>& f,
    const std::vector<Inpar::Solid::ModelType>* without_these_models) const
{
  if (not without_these_models) return assemble_force(timefac_np, f);

  Vector partial_me_vec;
  partial_me_vec.reserve(me_vec_ptr_->size());
  split_model_vector(partial_me_vec, *without_these_models);

  return assemble_force(partial_me_vec, timefac_np, f);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluatorManager::assemble_force(
    bool& ok, const Vector& me_vec, const double timefac_np, Core::LinAlg::Vector<double>& f) const
{
  if (not ok) return;

  for (const auto& cit : me_vec)
    // if one model evaluator failed, skip the remaining ones and return false
    ok = (ok ? cit->assemble_force(f, timefac_np) : false);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Solid::ModelEvaluatorManager::assemble_jacobian(const double timefac_np,
    Core::LinAlg::SparseOperator& jac,
    const std::vector<Inpar::Solid::ModelType>* without_these_models) const
{
  if (not without_these_models) return assemble_jacobian(timefac_np, jac);

  Vector partial_me_vec;
  split_model_vector(partial_me_vec, *without_these_models);

  return assemble_jacobian(partial_me_vec, timefac_np, jac);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluatorManager::assemble_jacobian(bool& ok, const Vector& me_vec,
    const double timefac_np, Core::LinAlg::SparseOperator& jac) const
{
  if (not ok) return;

  for (const auto& cit : me_vec)
    // if one model evaluator failed, skip the remaining ones and return false
    ok = (ok ? cit->assemble_jacobian(jac, timefac_np) : false);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluatorManager::assemble_jacobian_contributions_from_element_level_for_ptc(
    const Vector& me_vec, const double timefac_np,
    std::shared_ptr<Core::LinAlg::SparseMatrix>& modjac)
{
  for (const auto& cit : me_vec)
    cit->assemble_jacobian_contributions_from_element_level_for_ptc(modjac, timefac_np);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluatorManager::evaluate_force(bool& ok, const Vector& me_vec) const
{
  if (not ok) return;

  pre_evaluate(ok, me_vec);

  for (const auto& cit : me_vec)
    // if one model evaluator failed, skip the remaining ones and return false
    ok = (ok ? cit->evaluate_force() : false);

  post_evaluate(ok, me_vec);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluatorManager::evaluate_stiff(bool& ok, const Vector& me_vec) const
{
  if (not ok) return;

  pre_evaluate(ok, me_vec);

  for (const auto& cit : me_vec)
    // if one model evaluator failed, skip the remaining ones and return false
    ok = (ok ? cit->evaluate_stiff() : false);

  post_evaluate(ok, me_vec);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluatorManager::evaluate_force_stiff(bool& ok, const Vector& me_vec) const
{
  if (not ok) return;

  pre_evaluate(ok, me_vec);


  for (const auto& cit : me_vec)
    // if one model evaluator failed, skip the remaining ones and return false
    ok = (ok ? cit->evaluate_force_stiff() : false);

  post_evaluate(ok, me_vec);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluatorManager::pre_evaluate(bool ok, const Vector& me_vec) const
{
  for (const auto& me : me_vec) me->pre_evaluate();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluatorManager::post_evaluate(bool ok, const Vector& me_vec) const
{
  if (not ok) return;

  for (const auto& cit : me_vec) cit->post_evaluate();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Solid::ModelEvaluatorManager::apply_initial_force(
    const Core::LinAlg::Vector<double>& x, Core::LinAlg::Vector<double>& f)
{
  check_init_setup();

  bool ok = true;
  // initialize right hand side to zero
  f.put_scalar(0.0);

  // ---------------------------------------------------------------------------
  // reset model specific variables
  // ---------------------------------------------------------------------------
  reset_states(x, false);

  // ---------------------------------------------------------------------------
  // evaluate all terms
  // ---------------------------------------------------------------------------
  for (const auto& me_iter : *me_vec_ptr_)
    // if one model evaluator failed, skip the remaining ones and return false
    ok = (ok ? me_iter->evaluate_initial_force() : false);

  post_evaluate(ok, *me_vec_ptr_);

  // ---------------------------------------------------------------------------
  // put everything together, including mass and viscous contributions
  // ---------------------------------------------------------------------------
  assemble_force(ok, *me_vec_ptr_, 1.0, f);

  // ---------------------------------------------------------------------------
  // subtract mass and viscous contributions from initial force vector
  // ---------------------------------------------------------------------------
  f.scale(-1.);
  int_ptr_->add_visco_mass_contributions(f);
  f.scale(-1.);

  return ok;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluatorManager::reset_states(const Core::LinAlg::Vector<double>& x) const
{
  // default reset_states call
  reset_states(x, true);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluatorManager::reset_states(
    const Core::LinAlg::Vector<double>& x, bool setstate) const
{
  reset_states(x, setstate, *me_vec_ptr_);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluatorManager::reset_states(
    const Core::LinAlg::Vector<double>& x, bool setstate, Vector& me_vec) const
{
  if (setstate) int_ptr_->set_state(x);
  for (const auto& me_iter : me_vec) me_iter->reset(x);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Solid::ModelEvaluatorManager::apply_force(const Core::LinAlg::Vector<double>& x,
    Core::LinAlg::Vector<double>& f, const double& timefac_np) const
{
  check_init_setup();
  bool ok = true;
  // initialize right hand side to zero
  f.put_scalar(0.0);

  // ---------------------------------------------------------------------------
  // reset model specific variables
  // and also update the state variables of the current time integrator
  // ---------------------------------------------------------------------------
  reset_states(x);

  // ---------------------------------------------------------------------------
  // evaluate all terms
  // ---------------------------------------------------------------------------
  evaluate_force(ok, *me_vec_ptr_);

  // ---------------------------------------------------------------------------
  // put everything together
  // ---------------------------------------------------------------------------
  assemble_force(ok, *me_vec_ptr_, timefac_np, f);

  return ok;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Solid::ModelEvaluatorManager::apply_stiff(const Core::LinAlg::Vector<double>& x,
    Core::LinAlg::SparseOperator& jac, const double& timefac_np) const
{
  check_init_setup();
  bool ok = true;
  // initialize stiffness matrix to zero
  jac.zero();

  // ---------------------------------------------------------------------------
  // update the state variables of the current time integrator
  // reset model specific variables
  // ---------------------------------------------------------------------------
  reset_states(x);

  // ---------------------------------------------------------------------------
  // evaluate all terms
  // ---------------------------------------------------------------------------
  evaluate_stiff(ok, *me_vec_ptr_);

  // ---------------------------------------------------------------------------
  // put everything together
  // ---------------------------------------------------------------------------
  assemble_jacobian(ok, *me_vec_ptr_, timefac_np, jac);

  return ok;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Solid::ModelEvaluatorManager::apply_stiff(const Inpar::Solid::ModelType& mt,
    const Core::LinAlg::Vector<double>& x, Core::LinAlg::SparseOperator& jac,
    const double& timefac_np) const
{
  check_init_setup();
  bool ok = true;
  std::shared_ptr<Solid::ModelEvaluator::Generic> model_ptr = me_map_ptr_->at(mt);
  const Vector me_vec(1, model_ptr);

  // initialize stiffness matrix to zero
  jac.zero();

  // update the state variables of the current time integrator
  int_ptr_->set_state(x);
  model_ptr->reset(x);

  // ---------------------------------------------------------------------------
  // evaluate all terms
  // ---------------------------------------------------------------------------
  evaluate_stiff(ok, me_vec);

  // ---------------------------------------------------------------------------
  // put everything together
  // ---------------------------------------------------------------------------
  assemble_jacobian(ok, me_vec, timefac_np, jac);

  return ok;
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Solid::ModelEvaluatorManager::apply_force_stiff(const Core::LinAlg::Vector<double>& x,
    Core::LinAlg::Vector<double>& f, Core::LinAlg::SparseOperator& jac,
    const double& timefac_np) const
{
  check_init_setup();
  bool ok = true;
  // initialize stiffness matrix and right hand side to zero
  f.put_scalar(0.0);
  jac.zero();

  // ---------------------------------------------------------------------------
  // update the state variables of the current time integrator
  // reset model specific variables
  // ---------------------------------------------------------------------------
  reset_states(x);

  // ---------------------------------------------------------------------------
  // evaluate all terms
  // ---------------------------------------------------------------------------
  evaluate_force_stiff(ok, *me_vec_ptr_);

  // ---------------------------------------------------------------------------
  // put everything together
  // ---------------------------------------------------------------------------
  assemble_force(ok, *me_vec_ptr_, timefac_np, f);
  assemble_jacobian(ok, *me_vec_ptr_, timefac_np, jac);

  return ok;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Solid::ModelEvaluatorManager::apply_cheap_soc_rhs(const enum NOX::Nln::CorrectionType type,
    const std::vector<Inpar::Solid::ModelType>& constraint_models,
    const Core::LinAlg::Vector<double>& x, Core::LinAlg::Vector<double>& f,
    const double& timefac_np) const
{
  check_init_setup();

  Vector constraint_me_vec;
  extract_model_vector(constraint_me_vec, constraint_models);

  bool ok = true;
  // initialize right hand side to zero
  f.put_scalar(0.0);

  // ---------------------------------------------------------------------------
  // update the state variables of the current time integrator
  // reset model specific variables of the constraint models
  // ---------------------------------------------------------------------------
  reset_states(x, true, constraint_me_vec);

  // ---------------------------------------------------------------------------
  // evaluate all rhs terms of the constraint models
  // ---------------------------------------------------------------------------
  evaluate_cheap_soc_rhs(ok, constraint_me_vec);

  // ---------------------------------------------------------------------------
  // put everything together
  // ---------------------------------------------------------------------------
  assemble_cheap_soc_rhs(ok, constraint_me_vec, timefac_np, f);

  return ok;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluatorManager::evaluate_cheap_soc_rhs(bool& ok, const Vector& me_vec) const
{
  if (not ok) return;

  for (const auto& cit : me_vec)
    // if one model evaluator failed, skip the remaining ones and return false
    ok = (ok ? cit->evaluate_cheap_soc_rhs() : false);

  post_evaluate(ok, me_vec);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluatorManager::assemble_cheap_soc_rhs(
    bool& ok, const Vector& me_vec, const double timefac_np, Core::LinAlg::Vector<double>& f) const
{
  if (not ok) return;

  for (const auto& cit : me_vec)
    // if one model evaluator failed, skip the remaining ones and return false
    ok = (ok ? cit->assemble_cheap_soc_rhs(f, timefac_np) : false);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Solid::ModelEvaluatorManager::correct_parameters(
    const enum NOX::Nln::CorrectionType type) const
{
  bool ok = true;
  for (auto& cit : *me_vec_ptr_)
    // if one model evaluator failed, skip the remaining ones and return false
    ok = (ok ? cit->correct_parameters(type) : false);

  return ok;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluatorManager::post_setup()
{
  check_init();
  for (const auto& me_iter : *me_vec_ptr_) me_iter->post_setup();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluatorManager::predict(const Inpar::Solid::PredEnum& pred_type) const
{
  check_init_setup();
  for (const auto& me_iter : *me_vec_ptr_) me_iter->predict(pred_type);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluatorManager::write_restart(
    Core::IO::DiscretizationWriter& iowriter, const bool& forced_writerestart) const
{
  check_init_setup();
  for (const auto& me_iter : *me_vec_ptr_) me_iter->write_restart(iowriter, forced_writerestart);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluatorManager::read_restart(Core::IO::DiscretizationReader& ioreader)
{
  check_init_setup();
  for (const auto& me_iter : *me_vec_ptr_) me_iter->read_restart(ioreader);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluatorManager::run_recover() const
{
  check_init_setup();
  // set some parameters for the element evaluation
  eval_data_ptr_->set_is_default_step(true);
  eval_data_ptr_->set_step_length(1.0);
  eval_data_ptr_->reset_my_norms(true);
  for (const auto& me_iter : *me_vec_ptr_) me_iter->run_recover();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluatorManager::run_post_compute_x(const Core::LinAlg::Vector<double>& xold,
    const Core::LinAlg::Vector<double>& dir, const double& step,
    const Core::LinAlg::Vector<double>& xnew, const bool isdefaultstep) const
{
  check_init_setup();
  // set some parameters for the element evaluation
  eval_data_ptr_->set_is_default_step(isdefaultstep);
  eval_data_ptr_->set_step_length(step);
  eval_data_ptr_->reset_my_norms(isdefaultstep);
  for (const auto& me_iter : *me_vec_ptr_) me_iter->run_post_compute_x(xold, dir, xnew);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluatorManager::run_pre_compute_x(const Core::LinAlg::Vector<double>& xold,
    Core::LinAlg::Vector<double>& dir_mutable, const double& step, const NOX::Nln::Group& curr_grp,
    const bool isdefaultstep) const
{
  eval_data_ptr_->set_is_default_step(isdefaultstep);
  eval_data_ptr_->set_step_length(step);

  for (const auto& me_iter : *me_vec_ptr_) me_iter->run_pre_compute_x(xold, dir_mutable, curr_grp);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluatorManager::run_post_iterate(
    const ::NOX::Solver::Generic& solver, const double step, const bool isdefaultstep) const
{
  eval_data_ptr_->set_is_default_step(isdefaultstep);
  eval_data_ptr_->set_step_length(step);

  for (const auto& me_iter : *me_vec_ptr_) me_iter->run_post_iterate(solver);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluatorManager::run_pre_solve(
    const ::NOX::Solver::Generic& solver, const double step, const bool isdefaultstep) const
{
  eval_data_ptr_->set_is_default_step(isdefaultstep);
  eval_data_ptr_->set_step_length(step);

  for (const auto& me_iter : *me_vec_ptr_) me_iter->run_pre_solve(solver);
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluatorManager::run_post_apply_jacobian_inverse(
    const Core::LinAlg::Vector<double>& rhs, Core::LinAlg::Vector<double>& result,
    const Core::LinAlg::Vector<double>& xold, const NOX::Nln::Group& grp) const
{
  for (const auto& me_iter : *me_vec_ptr_)
    me_iter->run_post_apply_jacobian_inverse(rhs, result, xold, grp);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluatorManager::run_pre_apply_jacobian_inverse(
    const Core::LinAlg::Vector<double>& rhs, Core::LinAlg::Vector<double>& result,
    const Core::LinAlg::Vector<double>& xold, const NOX::Nln::Group& grp) const
{
  for (const auto& me_iter : *me_vec_ptr_)
    me_iter->run_pre_apply_jacobian_inverse(rhs, result, xold, grp);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const Solid::TimeInt::BaseDataGlobalState& Solid::ModelEvaluatorManager::get_global_state() const
{
  check_init_setup();
  return *gstate_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const std::shared_ptr<Solid::TimeInt::BaseDataGlobalState>&
Solid::ModelEvaluatorManager::global_state_ptr()
{
  check_init_setup();
  return gstate_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const std::shared_ptr<const Solid::TimeInt::Base>& Solid::ModelEvaluatorManager::get_tim_int_ptr()
    const
{
  check_init_setup();
  return timint_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Solid::ModelEvaluator::Generic& Solid::ModelEvaluatorManager::evaluator(
    const enum Inpar::Solid::ModelType& mt)
{
  check_init_setup();
  // sanity check, if there is a model evaluator for the given model type
  Solid::ModelEvaluatorManager::Map::const_iterator me_iter = me_map_ptr_->find(mt);
  if (me_iter == me_map_ptr_->end())
    FOUR_C_THROW("There is no model evaluator for the model type {}", mt);

  return *(me_iter->second);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const Solid::ModelEvaluator::Generic& Solid::ModelEvaluatorManager::evaluator(
    const enum Inpar::Solid::ModelType& mt) const
{
  check_init_setup();
  // sanity check, if there is a model evaluator for the given model type
  Solid::ModelEvaluatorManager::Map::const_iterator me_iter = me_map_ptr_->find(mt);
  if (me_iter == me_map_ptr_->end())
    FOUR_C_THROW("There is no model evaluator for the model type {}", mt);

  return *(me_iter->second);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluatorManager::update_step_state(const double& timefac_n)
{
  check_init_setup();
  /* Reset old structural right hand side.
   * It will be filled within the model evaluators */
  gstate_ptr_->get_fstructure_old()->put_scalar(0.0);
  for (const auto& me_iter : *me_vec_ptr_) me_iter->update_step_state(timefac_n);
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluatorManager::compute_jacobian_contributions_from_element_level_for_ptc(
    std::shared_ptr<Core::LinAlg::SparseMatrix>& scalingMatrixOpPtr)
{
  // evaluate ptc contributions at t^n+1
  double timefac_np = 1.0;
  for (const auto& me_iter : *me_vec_ptr_)
    me_iter->evaluate_jacobian_contributions_from_element_level_for_ptc();

  assemble_jacobian_contributions_from_element_level_for_ptc(
      *me_vec_ptr_, timefac_np, scalingMatrixOpPtr);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluatorManager::remove_condensed_contributions_from_rhs(
    Core::LinAlg::Vector<double>& rhs) const
{
  check_init_setup();
  for (const auto& me : *me_vec_ptr_) me->remove_condensed_contributions_from_rhs(rhs);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluatorManager::update_step_element()
{
  check_init_setup();
  for (const auto& me_iter : *me_vec_ptr_) me_iter->update_step_element();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluatorManager::update_residual()
{
  check_init_setup();
  for (const auto& me_iter : *me_vec_ptr_) me_iter->update_residual();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluatorManager::determine_stress_strain()
{
  check_init_setup();
  for (const auto& me_iter : *me_vec_ptr_) me_iter->determine_stress_strain();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluatorManager::determine_energy()
{
  check_init_setup();
  for (const auto& me_iter : *me_vec_ptr_) me_iter->determine_energy();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluatorManager::determine_optional_quantity()
{
  check_init_setup();
  for (const auto& me_iter : *me_vec_ptr_) me_iter->determine_optional_quantity();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluatorManager::output_step_state(Core::IO::DiscretizationWriter& iowriter) const
{
  check_init_setup();
  for (const auto& me_iter : *me_vec_ptr_) me_iter->output_step_state(iowriter);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluatorManager::runtime_pre_output_step_state()
{
  check_init_setup();
  for (const auto& me : *me_vec_ptr_) me->runtime_pre_output_step_state();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluatorManager::runtime_output_step_state() const
{
  check_init_setup();
  for (const auto& me_iter : *me_vec_ptr_) me_iter->runtime_output_step_state();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluatorManager::post_output()
{
  check_init_setup();
  for (const auto& me_iter : *me_vec_ptr_) me_iter->post_output();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluatorManager::reset_step_state()
{
  check_init_setup();
  for (const auto& me_iter : *me_vec_ptr_) me_iter->reset_step_state();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluatorManager::post_time_loop()
{
  check_init_setup();
  for (const auto& me_iter : *me_vec_ptr_) me_iter->post_time_loop();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<Solid::ModelEvaluatorManager::Vector>
Solid::ModelEvaluatorManager::transform_to_vector(
    const Solid::ModelEvaluatorManager::Map& model_map) const
{
  std::shared_ptr<Solid::ModelEvaluatorManager::Vector> me_vec_ptr =
      std::make_shared<Solid::ModelEvaluatorManager::Vector>(0);
  me_vec_ptr->reserve(model_map.size());

  // --------------------------------------------------------------------------
  // There must be a structural model evaluator at the first position
  // --------------------------------------------------------------------------
  if (model_map.begin()->first != Inpar::Solid::model_structure)
    FOUR_C_THROW(
        "The first model evaluator in the model_map must be a "
        "structural model evaluator!");

  // --------------------------------------------------------------------------
  // Fill the vector
  // --------------------------------------------------------------------------
  for (Map::const_iterator cit = model_map.begin(); cit != model_map.end(); ++cit)
    me_vec_ptr->push_back(cit->second);

  return me_vec_ptr;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluatorManager::split_model_vector(
    Solid::ModelEvaluatorManager::Vector& partial_me_vec,
    const std::vector<Inpar::Solid::ModelType>& without_these_models) const
{
  partial_me_vec.reserve(me_vec_ptr_->size());
  for (Vector::const_iterator cit = me_vec_ptr_->begin(); cit != me_vec_ptr_->end(); ++cit)
  {
    const Solid::ModelEvaluator::Generic& model = **cit;

    auto citer = without_these_models.cbegin();

    while (citer != without_these_models.cend())
    {
      if (*citer == model.type()) break;

      ++citer;
    }

    if (citer == without_these_models.cend()) partial_me_vec.push_back(*cit);
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluatorManager::extract_model_vector(
    Solid::ModelEvaluatorManager::Vector& partial_me_vec,
    const std::vector<Inpar::Solid::ModelType>& only_these_models) const
{
  partial_me_vec.reserve(only_these_models.size());
  for (const auto mtype : only_these_models)
  {
    auto cit = me_vec_ptr_->cbegin();
    while (cit != me_vec_ptr_->cend())
    {
      if ((*cit)->type() == mtype) break;

      ++cit;
    }

    FOUR_C_ASSERT(cit != me_vec_ptr_->end(), "Couldn't find the model type in me_vec_ptr_.");

    partial_me_vec.push_back(*cit);
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluatorManager::create_backup_state(const Core::LinAlg::Vector<double>& dir)
{
  check_init_setup();
  for (const auto& me_iter : *me_vec_ptr_) me_iter->create_backup_state(dir);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluatorManager::recover_from_backup_state()
{
  check_init_setup();
  for (const auto& me_iter : *me_vec_ptr_) me_iter->recover_from_backup_state();
}

FOUR_C_NAMESPACE_CLOSE
