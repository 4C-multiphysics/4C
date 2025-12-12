// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_structure_new_model_evaluator_contact.hpp"

#include "4C_contact_lagrange_strategy_poro.hpp"
#include "4C_contact_strategy_factory.hpp"
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_solver_nonlin_nox_group.hpp"
#include "4C_solver_nonlin_nox_solver_linesearchbased.hpp"
#include "4C_solver_nonlin_nox_vector.hpp"
#include "4C_structure_new_dbc.hpp"
#include "4C_structure_new_discretization_runtime_output_params.hpp"
#include "4C_structure_new_impl_generic.hpp"
#include "4C_structure_new_model_evaluator_data.hpp"
#include "4C_structure_new_model_evaluator_manager.hpp"
#include "4C_structure_new_timint_base.hpp"
#include "4C_structure_new_timint_basedataio_runtime_vtk_output.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Solid::ModelEvaluator::Contact::setup()
{
  check_init();
  eval_contact_ptr_ = eval_data().contact_ptr();

  // ---------------------------------------------------------------------
  // create the contact factory
  // ---------------------------------------------------------------------
  CONTACT::STRATEGY::Factory factory;
  factory.init(global_state_ptr()->get_discret());
  factory.setup(Global::Problem::instance()->n_dim());

  // check the problem dimension
  factory.check_dimension();

  // create some local variables (later to be stored in strategy)
  std::vector<std::shared_ptr<CONTACT::Interface>> interfaces;
  Teuchos::ParameterList cparams;

  // read and check contact input parameters
  factory.read_and_check_input(cparams);

  // check for fill_complete of discretization
  if (not discret().filled()) FOUR_C_THROW("discretization is not fillcomplete");

  // ---------------------------------------------------------------------
  // build the contact interfaces
  // ---------------------------------------------------------------------
  // FixMe Would be great, if we get rid of these poro parameters...
  bool poroslave = false;
  bool poromaster = false;
  factory.build_interfaces(cparams, interfaces, poroslave, poromaster);

  // ---------------------------------------------------------------------
  // build the solver strategy object
  // ---------------------------------------------------------------------
  strategy_ptr_ = factory.build_strategy(
      cparams, poroslave, poromaster, dof_offset(), interfaces, eval_contact_ptr_.get());

  // build the search tree
  factory.build_search_tree(interfaces);

  // print final screen output
  factory.print(interfaces, *strategy_ptr_, cparams);

  // ---------------------------------------------------------------------
  // final touches to the contact strategy
  // ---------------------------------------------------------------------
  strategy_ptr_->store_dirichlet_status(integrator().get_dbc().get_dbc_map_extractor());
  strategy_ptr_->set_state(Mortar::state_new_displacement, integrator().get_dbc().get_zeros());
  strategy_ptr_->save_reference_state(integrator().get_dbc().get_zeros_ptr());
  strategy_ptr_->evaluate_reference_state();
  strategy_ptr_->inttime_init();
  set_time_integration_info(*strategy_ptr_);
  strategy_ptr_->redistribute_contact(global_state().get_dis_n(), global_state().get_vel_n());

  check_pseudo2d();

  post_setup(cparams);

  if (global_in_output().get_runtime_output_params() != nullptr and
      global_in_output().get_runtime_output_params()->get_structure_params()->output_contact())
  {
    const auto discretization = this->discret_ptr();
    auto use_all_elements = [](const Core::Elements::Element* element) { return true; };
    contact_vtu_writer_ptr_ =
        std::make_unique<Core::IO::DiscretizationVisualizationWriterMesh>(discretization,
            Core::IO::visualization_parameters_factory(
                Global::Problem::instance()->io_params().sublist("RUNTIME VTK OUTPUT"),
                *Global::Problem::instance()->output_control_file(), global_state().get_time_n()),
            use_all_elements, discretization->name() + "-contact");
  }

  issetup_ = true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Solid::ModelEvaluator::Contact::post_setup(Teuchos::ParameterList& cparams)
{
  // do nothing
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Solid::ModelEvaluator::Contact::check_pseudo2d() const
{
  // print messages for multifield problems (e.g FSI)
  const Core::ProblemType probtype = Global::Problem::instance()->get_problem_type();
  if ((probtype != Core::ProblemType::structure) and (global_state().get_my_rank() == 0))
  {
    // warnings
#ifdef CONTACTPSEUDO2D
    std::cout << "WARNING: The flag CONTACTPSEUDO2D is switched on. If this "
              << "is a real 3D problem, switch it off!" << std::endl;
#else
    std::cout << "Solid::ModelEvaluator::Contact::check_pseudo2d -- "
              << "WARNING: \nThe flag CONTACTPSEUDO2D is switched off. If this "
              << "is a 2D problem modeled pseudo-3D, switch it on!" << std::endl;
#endif
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Solid::ModelEvaluator::Contact::set_time_integration_info(
    CONTACT::AbstractStrategy& strategy) const
{
  const Inpar::Solid::DynamicType dyntype = tim_int().get_data_sdyn().get_dynamic_type();
  const double time_fac = integrator().get_int_param();

  strategy.set_time_integration_info(time_fac, dyntype);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Solid::ModelEvaluator::Contact::reset(const Core::LinAlg::Vector<double>& x)
{
  check_init_setup();
  std::vector<std::shared_ptr<const Core::LinAlg::Vector<double>>> eval_vec(2, nullptr);
  eval_vec[0] = global_state().get_dis_np();
  eval_vec[1] = Core::Utils::shared_ptr_from_ref(x);
  eval_contact().set_action_type(Mortar::eval_reset);

  // reset displacement state and lagrange multiplier values
  strategy().evaluate(eval_data().contact(), &eval_vec);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool Solid::ModelEvaluator::Contact::evaluate_force()
{
  check_init_setup();
  bool ok = true;
  // --- evaluate contact contributions ---------------------------------
  eval_contact().set_action_type(Mortar::eval_force);
  eval_data().set_model_evaluator(this);
  strategy().evaluate(eval_data().contact());

  return ok;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool Solid::ModelEvaluator::Contact::evaluate_stiff()
{
  check_init_setup();
  bool ok = true;
  // --- evaluate contact contributions ---------------------------------
  eval_contact().set_action_type(Mortar::eval_force_stiff);
  eval_data().set_model_evaluator(this);
  strategy().evaluate(eval_data().contact());

  return ok;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool Solid::ModelEvaluator::Contact::evaluate_force_stiff()
{
  check_init_setup();
  bool ok = true;
  // --- evaluate contact contributions ---------------------------------
  eval_contact().set_action_type(Mortar::eval_force_stiff);
  strategy().evaluate(eval_data().contact());

  return ok;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Solid::ModelEvaluator::Contact::pre_evaluate()
{
  eval_contact().set_action_type(Mortar::eval_run_pre_evaluate);
  eval_data().set_model_evaluator(this);
  strategy().evaluate(eval_contact());
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Solid::ModelEvaluator::Contact::post_evaluate()
{
  eval_contact().set_action_type(Mortar::eval_run_post_evaluate);
  eval_data().set_model_evaluator(this);
  strategy().evaluate(eval_data().contact());
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool Solid::ModelEvaluator::Contact::assemble_force(
    Core::LinAlg::Vector<double>& f, const double& timefac_np) const
{
  std::shared_ptr<const Core::LinAlg::Vector<double>> block_vec_ptr = nullptr;

  if (Teuchos::getIntegralValue<Inpar::Mortar::AlgorithmType>(strategy().params(), "ALGORITHM") ==
          Inpar::Mortar::algorithm_gpts ||
      strategy().is_penalty() || strategy().is_condensed_system())
  {
    block_vec_ptr = strategy().get_rhs_block_ptr(CONTACT::VecBlockType::displ);

    // if there are no active contact contributions, we can skip this...
    if (!block_vec_ptr) return true;

    Core::LinAlg::assemble_my_vector(1.0, f, timefac_np, *block_vec_ptr);
  }
  else
  {
    // --- displ. - block ---------------------------------------------------
    block_vec_ptr = strategy().get_rhs_block_ptr(CONTACT::VecBlockType::displ);
    // if there are no active contact contributions, we can skip this...
    if (!block_vec_ptr) return true;
    Core::LinAlg::assemble_my_vector(1.0, f, timefac_np, *block_vec_ptr);

    // --- constr. - block --------------------------------------------------
    block_vec_ptr = strategy().get_rhs_block_ptr(CONTACT::VecBlockType::constraint);
    if (!block_vec_ptr) return true;
    Core::LinAlg::Vector<double> tmp(f.get_map());
    Core::LinAlg::export_to(*block_vec_ptr, tmp);
    f.update(1., tmp, 1.);
  }

  return true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool Solid::ModelEvaluator::Contact::assemble_jacobian(
    Core::LinAlg::SparseOperator& jac, const double& timefac_np) const
{
  std::shared_ptr<Core::LinAlg::SparseMatrix> block_ptr = nullptr;
  int err = 0;
  // ---------------------------------------------------------------------
  // Penalty / gpts / Nitsche system: no additional/condensed dofs
  // ---------------------------------------------------------------------
  if (Teuchos::getIntegralValue<Inpar::Mortar::AlgorithmType>(strategy().params(), "ALGORITHM") ==
          Inpar::Mortar::algorithm_gpts ||
      strategy().is_penalty())
  {
    block_ptr =
        strategy().get_matrix_block_ptr(CONTACT::MatBlockType::displ_displ, &eval_contact());
    if (strategy().is_penalty() && block_ptr == nullptr) return true;
    std::shared_ptr<Core::LinAlg::SparseMatrix> jac_dd = global_state().extract_displ_block(jac);
    Core::LinAlg::matrix_add(*block_ptr, false, timefac_np, *jac_dd, 1.0);
  }
  // ---------------------------------------------------------------------
  // condensed system of equations
  // ---------------------------------------------------------------------
  else if (strategy().is_condensed_system())
  {
    // --- Kdd - block ---------------------------------------------------
    block_ptr =
        strategy().get_matrix_block_ptr(CONTACT::MatBlockType::displ_displ, &eval_contact());
    if (block_ptr)
    {
      std::shared_ptr<Core::LinAlg::SparseMatrix> jac_dd_ptr =
          global_state().extract_displ_block(jac);
      Core::LinAlg::matrix_add(*block_ptr, false, timefac_np, *jac_dd_ptr, 1.0);
      // reset the block pointers, just to be on the safe side
      block_ptr = nullptr;
    }
  }
  // ---------------------------------------------------------------------
  // saddle-point system of equations or no contact contributions
  // ---------------------------------------------------------------------
  else if (strategy().system_type() == CONTACT::SystemType::saddlepoint)
  {
    // --- Kdd - block ---------------------------------------------------
    block_ptr =
        strategy().get_matrix_block_ptr(CONTACT::MatBlockType::displ_displ, &eval_contact());
    if (block_ptr)
    {
      std::shared_ptr<Core::LinAlg::SparseMatrix> jac_dd_ptr =
          global_state().extract_displ_block(jac);
      Core::LinAlg::matrix_add(*block_ptr, false, timefac_np, *jac_dd_ptr, 1.0);
      // reset the block pointers, just to be on the safe side
      block_ptr = nullptr;
    }

    // --- Kdz - block ---------------------------------------------------
    block_ptr = strategy().get_matrix_block_ptr(CONTACT::MatBlockType::displ_lm, &eval_contact());
    if (block_ptr)
    {
      block_ptr->scale(timefac_np);
      global_state().assign_model_block(jac, *block_ptr, type(), Solid::MatBlockType::displ_lm);
      // reset the block pointer, just to be on the safe side
      block_ptr = nullptr;
    }

    // --- Kzd - block ---------------------------------------------------
    block_ptr = strategy().get_matrix_block_ptr(CONTACT::MatBlockType::lm_displ, &eval_contact());
    if (block_ptr)
    {
      global_state().assign_model_block(jac, *block_ptr, type(), Solid::MatBlockType::lm_displ);
      // reset the block pointer, just to be on the safe side
      block_ptr = nullptr;
    }

    // --- Kzz - block ---------------------------------------------------
    block_ptr = strategy().get_matrix_block_ptr(CONTACT::MatBlockType::lm_lm, &eval_contact());
    if (block_ptr)
    {
      global_state().assign_model_block(jac, *block_ptr, type(), Solid::MatBlockType::lm_lm);
    }
    /* if there are no active contact contributions, we put a identity
     * matrix at the (lm,lm)-block */
    else
    {
      Core::LinAlg::Vector<double> ones(global_state().block_map(type()), false);
      ones.put_scalar(1.0);
      block_ptr = std::make_shared<Core::LinAlg::SparseMatrix>(ones);
      global_state().assign_model_block(jac, *block_ptr, type(), Solid::MatBlockType::lm_lm);
    }
    // reset the block pointer, just to be on the safe side
    block_ptr = nullptr;
  }

  return (err == 0);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Solid::ModelEvaluator::Contact::write_restart(
    Core::IO::DiscretizationWriter& iowriter, const bool& forced_writerestart) const
{
  // clear cache of maps due to varying vector size
  iowriter.clear_map_cache();

  // quantities to be written for restart
  std::map<std::string, std::shared_ptr<Core::LinAlg::Vector<double>>> restart_vectors;

  strategy().do_write_restart(restart_vectors, forced_writerestart);

  // write all vectors specified by used strategy
  std::map<std::string, std::shared_ptr<Core::LinAlg::Vector<double>>>::const_iterator p;
  for (p = restart_vectors.begin(); p != restart_vectors.end(); ++p)
    iowriter.write_vector(p->first, p->second);

  /* ToDo Move this stuff into the DoWriteRestart() routine of the
   * AbstractStrategy as soon as the old structural time integration
   * is gone! */
  if (strategy().lagrange_multiplier_n(true) != nullptr)
    iowriter.write_vector("lagrmultold", strategy().lagrange_multiplier_n(true));

  // since the global output_step_state() routine is not called, if the
  // restart is written, we have to do it here manually.
  output_step_state(iowriter);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Solid::ModelEvaluator::Contact::read_restart(Core::IO::DiscretizationReader& ioreader)
{
  eval_contact().set_action_type(Mortar::eval_force_stiff);
  // reader strategy specific stuff
  strategy().do_read_restart(ioreader, global_state().get_dis_n(), eval_data().contact_ptr());
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Solid::ModelEvaluator::Contact::update_step_state(const double& timefac_n)
{
  // add the contact forces to the old structural residual state vector
  std::shared_ptr<const Core::LinAlg::Vector<double>> strcontactrhs_ptr =
      strategy().get_rhs_block_ptr(CONTACT::VecBlockType::displ);
  if (strcontactrhs_ptr)
  {
    std::shared_ptr<Core::LinAlg::Vector<double>>& fstructold_ptr =
        global_state().get_fstructure_old();
    fstructold_ptr->update(timefac_n, *strcontactrhs_ptr, 1.0);
  }

  /* Note: DisN() and dis_np() have the same value at this stage, since
   * we call the structural model evaluator always in first place! */
  strategy_ptr_->update(global_state().get_dis_n());

  post_update_step_state();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Solid::ModelEvaluator::Contact::post_update_step_state()
{
  // initialize integration time for time measurement
  strategy_ptr_->inttime_init();

  // redistribute contact
  strategy_ptr_->redistribute_contact(global_state().get_dis_n(), global_state().get_vel_n());

  // setup the map extractor, since redistribute calls fill_complete
  // on the structural discretization. Though this only changes the
  // ghosted dofs while keeping row distribution fixed, the map pointers
  // in the global state are no longer valid. So we reset them.
  integrator().model_eval().setup_multi_map_extractor();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Solid::ModelEvaluator::Contact::update_step_element() { /* empty */ }

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Solid::ModelEvaluator::Contact::run_post_compute_x(const Core::LinAlg::Vector<double>& xold,
    const Core::LinAlg::Vector<double>& dir, const Core::LinAlg::Vector<double>& xnew)
{
  check_init_setup();

  std::vector<std::shared_ptr<const Core::LinAlg::Vector<double>>> eval_vec(3, nullptr);
  eval_vec[0] = Core::Utils::shared_ptr_from_ref(xold);
  eval_vec[1] = Core::Utils::shared_ptr_from_ref(dir);
  eval_vec[2] = Core::Utils::shared_ptr_from_ref(xnew);

  eval_contact().set_action_type(Mortar::eval_run_post_compute_x);

  strategy().evaluate(eval_data().contact(), &eval_vec);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Solid::ModelEvaluator::Contact::determine_stress_strain()
{
  // evaluate contact tractions
  strategy().compute_contact_stresses();

  if (strategy().weighted_wear())
  {
    /* *******************************************************************
     * We do not compute the non-weighted wear here. we just write
     * the output. the non-weighted wear will be used as dirichlet-b.
     * for the ale problem. n.w.wear will be called in
     * stru_ale_algorithm.cpp and computed in strategy.OutputWear();
     *                                                         farah 06/13
     * ******************************************************************/
    // evaluate wear (not weighted)
    strategy().reset_wear();
    strategy().output_wear();
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Solid::ModelEvaluator::Contact::determine_energy() {}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Solid::ModelEvaluator::Contact::runtime_pre_output_step_state() { determine_stress_strain(); }

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Solid::ModelEvaluator::Contact::runtime_output_step_state() const
{
  if (contact_vtu_writer_ptr_ == nullptr) return;

  // reset time and time step of the writer object
  contact_vtu_writer_ptr_->reset();

  // no output in nitsche Strategy
  if (strategy().is_nitsche()) return;

  // print summary of active set to screen
  strategy().print_active_set();

  // evaluate active set and slip set
  // TODO as soon as true int vectors are supported this should be changed to an int vector
  Core::LinAlg::Vector<double> activeset(*strategy().active_row_nodes(), true);
  activeset.put_scalar(1.0);
  if (strategy().is_friction())
  {
    Core::LinAlg::Vector<double> slipset(*strategy().slip_row_nodes(), true);
    slipset.put_scalar(1.0);
    Core::LinAlg::Vector<double> slipsetexp(*strategy().active_row_nodes(), true);
    Core::LinAlg::export_to(slipset, slipsetexp);
    activeset.update(1.0, slipsetexp, 1.0);
  }

  // export to problem node row map
  auto problemnodes = strategy().problem_nodes();
  auto activesetexp = Core::LinAlg::Vector<double>(*problemnodes, true);
  Core::LinAlg::export_to(activeset, activesetexp);

  if (strategy().wear_both_discrete())
  {
    Core::LinAlg::Vector<double> mactiveset(*strategy().master_active_nodes(), true);
    mactiveset.put_scalar(1.0);
    Core::LinAlg::Vector<double> slipset(*strategy().master_slip_nodes(), true);
    slipset.put_scalar(1.0);
    Core::LinAlg::Vector<double> slipsetexp(*strategy().master_active_nodes(), true);
    Core::LinAlg::export_to(slipset, slipsetexp);
    mactiveset.update(1.0, slipsetexp, 1.0);

    Core::LinAlg::Vector<double> mactivesetexp(*problemnodes, true);
    Core::LinAlg::export_to(mactiveset, mactivesetexp);
    activesetexp.update(1.0, mactivesetexp, 1.0);
  }

  contact_vtu_writer_ptr_->append_result_data_vector_with_context(
      activesetexp, Core::IO::OutputEntity::node, {"activeset"});

  // contact tractions
  // export to problem dof row map
  auto problemdofs = strategy().problem_dofs();

  // normal direction
  auto normalstresses = strategy().contact_normal_stress();
  auto normalstressesexp = Core::LinAlg::Vector<double>(*problemdofs, true);
  Core::LinAlg::export_to(*normalstresses, normalstressesexp);

  // tangential plane
  auto tangentialstresses = strategy().contact_tangential_stress();
  auto tangentialstressesexp = Core::LinAlg::Vector<double>(*problemdofs, true);
  Core::LinAlg::export_to(*tangentialstresses, tangentialstressesexp);

  // write to output
  // contact tractions in normal and tangential direction
  {
    std::vector<std::optional<std::string>> context(global_state().get_dim(), "norcontactstress");
    contact_vtu_writer_ptr_->append_result_data_vector_with_context(
        normalstressesexp, Core::IO::OutputEntity::dof, context);
  }
  {
    std::vector<std::optional<std::string>> context(global_state().get_dim(), "tancontactstress");
    contact_vtu_writer_ptr_->append_result_data_vector_with_context(
        tangentialstressesexp, Core::IO::OutputEntity::dof, context);
  }

  // wear with internal state variable approach
  if (strategy().weighted_wear())
  {
    // write output
    auto wearoutput = strategy().contact_wear();
    auto wearoutputexp = Core::LinAlg::Vector<double>(*problemdofs, true);
    Core::LinAlg::export_to(*wearoutput, wearoutputexp);

    {
      std::vector<std::optional<std::string>> context(global_state().get_dim(), "wear");
      contact_vtu_writer_ptr_->append_result_data_vector_with_context(
          wearoutputexp, Core::IO::OutputEntity::dof, context);
    }
  }

  // poro contact
  if (strategy().has_poro_no_penetration())
  {
    // output of poro no penetration lagrange multiplier!
    auto lambdaout = dynamic_cast<const CONTACT::LagrangeStrategyPoro&>(strategy()).lambda_no_pen();
    auto lambdaoutexp = Core::LinAlg::Vector<double>(*problemdofs, true);
    Core::LinAlg::export_to(*lambdaout, lambdaoutexp);

    {
      std::vector<std::optional<std::string>> context(global_state().get_dim(), "poronopen_lambda");
      contact_vtu_writer_ptr_->append_result_data_vector_with_context(
          lambdaoutexp, Core::IO::OutputEntity::dof, context);
    }
  }

  contact_vtu_writer_ptr_->write_to_disk(global_state().get_time_n(), global_state().get_step_n());
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Solid::ModelEvaluator::Contact::reset_step_state()
{
  check_init_setup();

  FOUR_C_THROW("Not yet implemented");
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Solid::ModelEvaluator::ContactData& Solid::ModelEvaluator::Contact::eval_contact()
{
  check_init_setup();
  return *eval_contact_ptr_;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
const Solid::ModelEvaluator::ContactData& Solid::ModelEvaluator::Contact::eval_contact() const
{
  check_init_setup();
  return *eval_contact_ptr_;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
const std::shared_ptr<CONTACT::AbstractStrategy>& Solid::ModelEvaluator::Contact::strategy_ptr()
{
  check_init_setup();
  return strategy_ptr_;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
CONTACT::AbstractStrategy& Solid::ModelEvaluator::Contact::strategy()
{
  check_init_setup();
  return *strategy_ptr_;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
const CONTACT::AbstractStrategy& Solid::ModelEvaluator::Contact::strategy() const
{
  check_init_setup();
  return *strategy_ptr_;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Map> Solid::ModelEvaluator::Contact::get_block_dof_row_map_ptr()
    const
{
  Global::Problem* problem = Global::Problem::instance();

  check_init_setup();
  if (strategy().lm_dof_row_map_ptr(false) == nullptr)
    return global_state().dof_row_map();
  else
  {
    auto systype =
        Teuchos::getIntegralValue<CONTACT::SystemType>(problem->contact_dynamic_params(), "SYSTEM");

    if (systype == CONTACT::SystemType::saddlepoint)
      return strategy().lin_system_lm_dof_row_map_ptr();
    else
      return global_state().dof_row_map();
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Vector<double>>
Solid::ModelEvaluator::Contact::get_current_solution_ptr() const
{
  // TODO: this should be removed!
  Global::Problem* problem = Global::Problem::instance();
  auto systype =
      Teuchos::getIntegralValue<CONTACT::SystemType>(problem->contact_dynamic_params(), "SYSTEM");
  if (systype == CONTACT::SystemType::condensed) return nullptr;

  if (strategy().lagrange_multiplier_np(false) != nullptr)
  {
    std::shared_ptr<Core::LinAlg::Vector<double>> curr_lm_ptr =
        std::make_shared<Core::LinAlg::Vector<double>>(*strategy().lagrange_multiplier_np(false));
    if (curr_lm_ptr) curr_lm_ptr->replace_map(strategy().lm_dof_row_map(false));

    extend_lagrange_multiplier_domain(curr_lm_ptr);

    return curr_lm_ptr;
  }
  else
    return nullptr;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Vector<double>>
Solid::ModelEvaluator::Contact::get_last_time_step_solution_ptr() const
{
  Global::Problem* problem = Global::Problem::instance();
  auto systype =
      Teuchos::getIntegralValue<CONTACT::SystemType>(problem->contact_dynamic_params(), "SYSTEM");
  if (systype == CONTACT::SystemType::condensed) return nullptr;

  if (strategy().lagrange_multiplier_n(false) == nullptr) return nullptr;

  std::shared_ptr<Core::LinAlg::Vector<double>> old_lm_ptr =
      std::make_shared<Core::LinAlg::Vector<double>>(*strategy().lagrange_multiplier_n(false));
  if (old_lm_ptr) old_lm_ptr->replace_map(strategy().lm_dof_row_map(false));

  extend_lagrange_multiplier_domain(old_lm_ptr);

  return old_lm_ptr;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Solid::ModelEvaluator::Contact::extend_lagrange_multiplier_domain(
    std::shared_ptr<Core::LinAlg::Vector<double>>& lm_vec) const
{
  // default case: do nothing
  if (strategy().lm_dof_row_map(false).num_global_elements() ==
      get_block_dof_row_map_ptr()->num_global_elements())
    return;

  if (strategy().lm_dof_row_map(false).num_global_elements() <
      get_block_dof_row_map_ptr()->num_global_elements())
  {
    std::shared_ptr<Core::LinAlg::Vector<double>> tmp_ptr =
        std::make_shared<Core::LinAlg::Vector<double>>(*get_block_dof_row_map_ptr());
    Core::LinAlg::export_to(*lm_vec, *tmp_ptr);
    lm_vec = tmp_ptr;
  }
  else
    FOUR_C_THROW("Unconsidered case.");
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Solid::ModelEvaluator::Contact::post_output()
{
  check_init_setup();
  // empty
}  // post_output()

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::Contact::run_pre_compute_x(const Core::LinAlg::Vector<double>& xold,
    Core::LinAlg::Vector<double>& dir_mutable, const NOX::Nln::Group& curr_grp)
{
  check_init_setup();

  std::vector<std::shared_ptr<const Core::LinAlg::Vector<double>>> eval_vec(1, nullptr);
  eval_vec[0] = Core::Utils::shared_ptr_from_ref(xold);

  std::vector<std::shared_ptr<Core::LinAlg::Vector<double>>> eval_vec_mutable(1, nullptr);
  eval_vec_mutable[0] = Core::Utils::shared_ptr_from_ref(dir_mutable);

  eval_contact().set_action_type(Mortar::eval_run_pre_compute_x);
  eval_data().set_model_evaluator(this);

  // augment the search direction
  strategy().evaluate(eval_data().contact(), &eval_vec, &eval_vec_mutable);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::Contact::run_post_iterate(const ::NOX::Solver::Generic& solver)
{
  check_init_setup();

  const auto& nox_x = dynamic_cast<const NOX::Nln::Vector&>(solver.getSolutionGroup().getX());

  // displacement vector after the predictor call
  std::shared_ptr<Core::LinAlg::Vector<double>> curr_disp =
      global_state().extract_displ_entries(nox_x.get_linalg_vector());
  std::shared_ptr<const Core::LinAlg::Vector<double>> curr_vel = global_state().get_vel_np();

  if (strategy().dyn_redistribute_contact(curr_disp, curr_vel, solver.getNumIterations()))
    evaluate_force();

  eval_contact().set_action_type(Mortar::eval_run_post_iterate);
  strategy().evaluate(eval_data().contact());
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::Contact::run_pre_apply_jacobian_inverse(
    const Core::LinAlg::Vector<double>& rhs, Core::LinAlg::Vector<double>& result,
    const Core::LinAlg::Vector<double>& xold, const NOX::Nln::Group& grp)
{
  std::shared_ptr<Core::LinAlg::SparseMatrix> jac_dd = global_state().jacobian_displ_block();
  const_cast<CONTACT::AbstractStrategy&>(strategy())
      .run_pre_apply_jacobian_inverse(jac_dd, const_cast<Core::LinAlg::Vector<double>&>(rhs));
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::Contact::run_pre_solve(const ::NOX::Solver::Generic& solver)
{
  check_init_setup();
  const auto& nox_x = dynamic_cast<const NOX::Nln::Vector&>(solver.getSolutionGroup().getX());

  // displacement vector after the predictor call
  std::shared_ptr<Core::LinAlg::Vector<double>> curr_disp =
      global_state().extract_displ_entries(nox_x.get_linalg_vector());

  std::vector<std::shared_ptr<const Core::LinAlg::Vector<double>>> eval_vec(1, nullptr);
  eval_vec[0] = curr_disp;

  eval_contact().set_action_type(Mortar::eval_run_pre_solve);
  strategy().evaluate(eval_data().contact(), &eval_vec);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::Contact::run_post_apply_jacobian_inverse(
    const Core::LinAlg::Vector<double>& rhs, Core::LinAlg::Vector<double>& result,
    const Core::LinAlg::Vector<double>& xold, const NOX::Nln::Group& grp)
{
  check_init_setup();

  CONTACT::AbstractStrategy::PostApplyJacobianData data{
      .rhs = &rhs,
      .result = &result,
      .xold = &xold,
      .grp = &grp,
  };
  eval_contact().set_user_data(std::any{data});

  eval_contact().set_action_type(Mortar::eval_run_post_apply_jacobian_inverse);
  eval_data().set_model_evaluator(this);

  // augment the search direction
  strategy().evaluate(eval_data().contact());
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::SparseMatrix>
Solid::ModelEvaluator::Contact::get_jacobian_block(const MatBlockType bt) const
{
  return global_state().get_jacobian_block(type(), bt);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>>
Solid::ModelEvaluator::Contact::assemble_force_of_models(
    const std::vector<Inpar::Solid::ModelType>* without_these_models, const bool apply_dbc) const
{
  std::shared_ptr<NOX::Nln::Vector> force_nox = global_state().create_global_vector();
  integrator().assemble_force(force_nox->get_linalg_vector(), without_these_models);

  // copy the vector, otherwise the storage will be freed at the end of this
  // function, resulting in a segmentation fault
  std::shared_ptr<Core::LinAlg::Vector<double>> force =
      std::make_shared<Core::LinAlg::Vector<double>>(force_nox->get_linalg_vector());

  if (apply_dbc) tim_int().get_dbc().apply_dirichlet_to_rhs(*force);

  return force;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::SparseOperator>
Solid::ModelEvaluator::Contact::get_aux_displ_jacobian() const
{
  std::vector<Inpar::Solid::ModelType> g;
  g.push_back(Inpar::Solid::ModelType::model_contact);

  std::shared_ptr<Core::LinAlg::SparseOperator> jacaux = global_state().create_aux_jacobian();
  bool ok = integrator().assemble_jac(*jacaux, &g);

  if (!ok) FOUR_C_THROW("ERROR: create_aux_jacobian went wrong!");

  return jacaux;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::Contact::evaluate_weighted_gap_gradient_error()
{
  eval_contact().set_action_type(Mortar::eval_wgap_gradient_error);
  eval_data().set_model_evaluator(this);

  // augment the search direction
  strategy().evaluate(eval_data().contact());
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Solid::ModelEvaluator::Contact::evaluate_cheap_soc_rhs()
{
  check_init_setup();

  eval_contact().set_action_type(Mortar::eval_static_constraint_rhs);
  eval_data().set_model_evaluator(this);

  strategy().evaluate(eval_data().contact());

  return true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Solid::ModelEvaluator::Contact::assemble_cheap_soc_rhs(
    Core::LinAlg::Vector<double>& f, const double& timefac_np) const
{
  return assemble_force(f, timefac_np);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::Contact::remove_condensed_contributions_from_rhs(
    Core::LinAlg::Vector<double>& rhs)
{
  check_init_setup();
  eval_contact().set_action_type(Mortar::remove_condensed_contributions_from_str_rhs);

  std::vector<std::shared_ptr<Core::LinAlg::Vector<double>>> mutable_vec(
      1, Core::Utils::shared_ptr_from_ref(rhs));
  strategy().evaluate(eval_contact(), nullptr, &mutable_vec);
}

FOUR_C_NAMESPACE_CLOSE
