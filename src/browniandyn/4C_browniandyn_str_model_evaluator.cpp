// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_browniandyn_str_model_evaluator.hpp"

#include "4C_beam3_base.hpp"
#include "4C_beaminteraction_calc_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_geometry_periodic_boundingbox.hpp"
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_rigidsphere.hpp"
#include "4C_structure_new_integrator.hpp"
#include "4C_structure_new_model_evaluator_data.hpp"
#include "4C_structure_new_timint_base.hpp"

#include <Teuchos_ParameterList.hpp>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Solid::ModelEvaluator::BrownianDyn::BrownianDyn()
    : eval_browniandyn_ptr_(nullptr),
      f_brown_np_ptr_(nullptr),
      f_ext_np_ptr_(nullptr),
      stiff_brownian_ptr_(nullptr),
      maxrandnumelement_(0),
      discret_ptr_(nullptr)
{
  // empty
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::BrownianDyn::setup()
{
  check_init();

  // safety check, brownian dynamics simulation only for one step theta and
  // theta = 1.0 (see Cyron 2012)
  if (tim_int().get_data_sdyn_ptr()->get_dynamic_type() != Inpar::Solid::DynamicType::OneStepTheta)
    FOUR_C_THROW("Brownian dynamics simulation only consistent for one step theta schema.");

  discret_ptr_ = discret_ptr();

  // -------------------------------------------------------------------------
  // get pointer to biopolymer network data and init random number data
  // -------------------------------------------------------------------------
  eval_browniandyn_ptr_ = eval_data().brownian_dyn_ptr();
  brown_dyn_state_data_.browndyn_dt = eval_browniandyn_ptr_->time_step_const_rand_numb();

  // todo: maybe make input of time step obligatory
  if (brown_dyn_state_data_.browndyn_dt < 0.0)
  {
    brown_dyn_state_data_.browndyn_dt = global_state().get_delta_time()[0];
    if (global_state().get_my_rank() == 0)
      std::cout << " Time step " << global_state().get_delta_time()[0]
                << " form Structural Dynamic section used for stochastic forces.\n"
                << std::endl;
  }

  brown_dyn_state_data_.browndyn_step = -1;
  // -------------------------------------------------------------------------
  // setup the brownian forces and the external force pointers
  // -------------------------------------------------------------------------
  f_brown_np_ptr_ =
      std::make_shared<Core::LinAlg::Vector<double>>(*global_state().dof_row_map(), true);
  f_ext_np_ptr_ =
      std::make_shared<Core::LinAlg::Vector<double>>(*global_state().dof_row_map(), true);
  // -------------------------------------------------------------------------
  // setup the brownian forces and the external force pointers
  // -------------------------------------------------------------------------
  stiff_brownian_ptr_ = std::make_shared<Core::LinAlg::SparseMatrix>(
      *global_state().dof_row_map_view(), 81, true, true);

  // -------------------------------------------------------------------------
  // get maximal number of random numbers required by any element in the
  // discretization and store them in randomnumbersperelement_
  // -------------------------------------------------------------------------
  random_numbers_per_element();
  // -------------------------------------------------------------------------
  // Generate random forces for first time step
  // -------------------------------------------------------------------------
  /* multivector for stochastic forces evaluated by each element; the numbers of
   * vectors in the multivector equals the maximal number of random numbers
   * required by any element in the discretization per time step; therefore this
   * multivector is suitable for synchronization of these random numbers in
   *  parallel computing*/
  eval_browniandyn_ptr_->resize_random_force_m_vector(discret_ptr_, maxrandnumelement_);
  generate_gaussian_random_numbers();

  issetup_ = true;

  return;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::BrownianDyn::reset(const Core::LinAlg::Vector<double>& x)
{
  check_init_setup();

  // todo: somewhat illegal considering of const correctness
  tim_int().get_data_sdyn_ptr()->get_periodic_bounding_box()->apply_dirichlet(
      global_state().get_time_n(), Global::Problem::instance()->function_manager());

  // -------------------------------------------------------------------------
  // reset brownian (stochastic and damping) forces
  // -------------------------------------------------------------------------
  f_brown_np_ptr_->put_scalar(0.0);
  // -------------------------------------------------------------------------
  // reset external forces
  // -------------------------------------------------------------------------
  f_ext_np_ptr_->put_scalar(0.0);
  // -------------------------------------------------------------------------
  // zero out brownian stiffness contributions
  // -------------------------------------------------------------------------
  stiff_brownian_ptr_->zero();

  return;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Solid::ModelEvaluator::BrownianDyn::evaluate_force()
{
  check_init_setup();
  bool ok = true;
  // ---------------------------------------
  // (1) EXTERNAL FORCES
  // ---------------------------------------
  ok = apply_force_external();

  // ---------------------------------------
  // (2) INTERNAL FORCES
  // ---------------------------------------
  // ordinary internal force
  ok = (ok ? apply_force_brownian() : false);

  return ok;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Solid::ModelEvaluator::BrownianDyn::evaluate_stiff()
{
  check_init_setup();
  bool ok = true;

  /* We use the same routines as for the apply_force_stiff case, but we
   * do not update the global force vector, which is used for the
   * solution process in the NOX library.
   * This is meaningful, since the computational overhead, which is
   * generated by evaluating the right hand side is negligible */

  // -------------------------------------------------------------------------
  // (1) EXTRERNAL FORCES and STIFFNESS ENTRIES
  // -------------------------------------------------------------------------
  // so far the Neumann loads implemented especially for brownian don't
  // have a contribution to the jacobian
  //   apply_force_stiff_external();

  // -------------------------------------------------------------------------
  // (2) BROWNIAN FORCES and STIFFNESS ENTRIES
  // -------------------------------------------------------------------------
  apply_force_stiff_brownian();

  if (not stiff_brownian_ptr_->filled()) stiff_brownian_ptr_->complete();

  return ok;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Solid::ModelEvaluator::BrownianDyn::evaluate_force_stiff()
{
  check_init_setup();
  bool ok = true;

  // -------------------------------------------------------------------------
  // (1) EXTRERNAL FORCES and STIFFNESS ENTRIES
  // -------------------------------------------------------------------------
  apply_force_stiff_external();
  // -------------------------------------------------------------------------
  // (2) BROWNIAN FORCES and STIFFNESS ENTRIES
  // -------------------------------------------------------------------------
  apply_force_stiff_brownian();

  if (not stiff_brownian_ptr_->filled()) stiff_brownian_ptr_->complete();

  return ok;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Solid::ModelEvaluator::BrownianDyn::assemble_force(
    Core::LinAlg::Vector<double>& f, const double& timefac_np) const
{
  check_init_setup();

  // safety check, brownian dynamics simulation for with one step theta and
  // theta = 1.0 (see Cyron 2012)
  if (abs(timefac_np - 1.0) > 1.0e-8)
    FOUR_C_THROW(
        "Brownian dynamics simulation only consistent for one step theta scheme"
        " and theta = 1.0 .");

  // -------------------------------------------------------------------------
  // *********** finally put everything together ***********
  // build residual  Res = F_{brw;n+1}
  //                     - F_{ext;n+1}
  // -------------------------------------------------------------------------
  Core::LinAlg::assemble_my_vector(1.0, f, -timefac_np, *f_ext_np_ptr_);
  Core::LinAlg::assemble_my_vector(1.0, f, timefac_np, *f_brown_np_ptr_);

  return true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Solid::ModelEvaluator::BrownianDyn::assemble_jacobian(
    Core::LinAlg::SparseOperator& jac, const double& timefac_np) const
{
  check_init_setup();

  std::shared_ptr<Core::LinAlg::SparseMatrix> jac_dd_ptr = global_state().extract_displ_block(jac);
  Core::LinAlg::matrix_add(*stiff_brownian_ptr_, false, timefac_np, *jac_dd_ptr, 1.0);
  // no need to keep it
  stiff_brownian_ptr_->zero();

  return true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Solid::ModelEvaluator::BrownianDyn::apply_force_external()
{
  check_init_setup();
  bool ok = true;
  // -------------------------------------------------------------------------
  // Set to default value  as it is unnecessary for the
  // evaluate_neumann routine.
  // -------------------------------------------------------------------------
  eval_data().set_action_type(Core::Elements::none);
  // -------------------------------------------------------------------------
  // set vector values needed by elements
  // -------------------------------------------------------------------------
  discret().clear_state();
  discret().set_state(0, "displacement", *global_state().get_dis_n());
  // -------------------------------------------------------------------------
  // Evaluate brownian specific neumann conditions
  // -------------------------------------------------------------------------
  evaluate_neumann_brownian_dyn(f_ext_np_ptr_, nullptr);

  return ok;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Solid::ModelEvaluator::BrownianDyn::apply_force_brownian()
{
  check_init_setup();
  bool ok = true;
  // -------------------------------------------------------------------------
  // currently a fixed number of matrix and vector pointers are supported
  // set default matrices and vectors
  // -------------------------------------------------------------------------
  std::array<std::shared_ptr<Core::LinAlg::Vector<double>>, 3> eval_vec = {
      nullptr, nullptr, nullptr};
  std::array<std::shared_ptr<Core::LinAlg::SparseOperator>, 2> eval_mat = {nullptr, nullptr};
  // -------------------------------------------------------------------------
  // set brwonian force vector (gets filled on element level)
  // -------------------------------------------------------------------------
  eval_vec[0] = f_brown_np_ptr_;
  // -------------------------------------------------------------------------
  // set action for elements
  // -------------------------------------------------------------------------
  eval_data().set_action_type(Core::Elements::struct_calc_brownianforce);
  // -------------------------------------------------------------------------
  // set vector values needed by elements
  // -------------------------------------------------------------------------
  discret().clear_state();
  discret().set_state(0, "displacement", *global_state_ptr()->get_dis_np());
  discret().set_state(0, "velocity", *global_state().get_vel_np());
  // -------------------------------------------------------------------------
  // Evaluate Browian (stochastic and damping forces)
  // -------------------------------------------------------------------------
  evaluate_brownian(eval_mat.data(), eval_vec.data());

  return ok;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Solid::ModelEvaluator::BrownianDyn::apply_force_stiff_external()
{
  /* so far brownian specific neumann loads need no linearization,
   therefore apply_force_stiff_external is equal to apply_force_external*/

  check_init_setup();
  bool ok = true;
  // -------------------------------------------------------------------------
  // Set to default value, as it is unnecessary for the
  // evaluate_neumann routine.
  // -------------------------------------------------------------------------
  eval_data().set_action_type(Core::Elements::none);
  // -------------------------------------------------------------------------
  // set vector values needed by elements
  // -------------------------------------------------------------------------
  discret().clear_state();
  discret().set_state(0, "displacement", *global_state().get_dis_n());
  // -------------------------------------------------------------------------
  // Evaluate brownian specific neumann conditions
  // -------------------------------------------------------------------------
  evaluate_neumann_brownian_dyn(f_ext_np_ptr_, nullptr);

  return ok;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool Solid::ModelEvaluator::BrownianDyn::apply_force_stiff_brownian()
{
  check_init_setup();
  bool ok = true;
  // -------------------------------------------------------------------------
  // currently a fixed number of matrix and vector pointers are supported
  // set default matrices and vectors
  // -------------------------------------------------------------------------
  std::array<std::shared_ptr<Core::LinAlg::Vector<double>>, 3> eval_vec = {
      nullptr, nullptr, nullptr};
  std::array<std::shared_ptr<Core::LinAlg::SparseOperator>, 2> eval_mat = {nullptr, nullptr};
  // -------------------------------------------------------------------------
  // set jac matrix and brownian force vector (filled on element level)
  // -------------------------------------------------------------------------
  eval_mat[0] = stiff_brownian_ptr_;
  eval_vec[0] = f_brown_np_ptr_;
  // -------------------------------------------------------------------------
  // set action for elements
  // -------------------------------------------------------------------------
  eval_data().set_action_type(Core::Elements::struct_calc_brownianstiff);
  // -------------------------------------------------------------------------
  // set vector values needed by elements
  // -------------------------------------------------------------------------
  discret().clear_state();
  discret().set_state(0, "displacement", *global_state_ptr()->get_dis_np());
  discret().set_state(0, "velocity", *global_state().get_vel_np());
  // -------------------------------------------------------------------------
  // Evaluate brownian (stochastic and damping) forces
  evaluate_brownian(eval_mat.data(), eval_vec.data());

  return ok;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::BrownianDyn::evaluate_brownian(
    std::shared_ptr<Core::LinAlg::SparseOperator>* eval_mat,
    std::shared_ptr<Core::LinAlg::Vector<double>>* eval_vec)
{
  check_init_setup();

  // todo: just give params_interface to elements (not a parameter list)
  Teuchos::ParameterList p;
  p.set<std::shared_ptr<Core::Elements::ParamsInterface>>("interface", eval_data_ptr());
  // -------------------------------------------------------------------------
  // Evaluate brownian (stochastic and damping) forces on element level
  // -------------------------------------------------------------------------
  evaluate_brownian(p, eval_mat, eval_vec);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::BrownianDyn::evaluate_brownian(Teuchos::ParameterList& p,
    std::shared_ptr<Core::LinAlg::SparseOperator>* eval_mat,
    std::shared_ptr<Core::LinAlg::Vector<double>>* eval_vec)
{
  check_init_setup();

  // todo: this needs to go, just pass params_interface to elements
  if (p.numParams() > 1)
    FOUR_C_THROW(
        "Please use the Solid::Elements::Interface and its derived "
        "classes to set and get parameters.");
  // -------------------------------------------------------------------------
  // Evaluate brownian on element level
  // -------------------------------------------------------------------------
  discret().evaluate(p, eval_mat[0], eval_mat[1], eval_vec[0], eval_vec[1], eval_vec[2]);
  discret().clear_state();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::BrownianDyn::evaluate_neumann_brownian_dyn(
    std::shared_ptr<Core::LinAlg::Vector<double>> eval_vec,
    std::shared_ptr<Core::LinAlg::SparseOperator> eval_mat)
{
  (void)eval_vec;
  (void)eval_mat;

  check_init_setup();
  // -------------------------------------------------------------------------
  // get interface pointer
  // -------------------------------------------------------------------------
  std::shared_ptr<Core::Elements::ParamsInterface> interface_ptr = eval_data_ptr();
  // -------------------------------------------------------------------------
  // evaluate brownian specific Neumann boundary conditions
  // -------------------------------------------------------------------------
  //  sm_manager_ptr_->EvaluateNeumannbrownian(interface_ptr,eval_vec,eval_mat);
  discret_ptr()->clear_state();

  return;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::BrownianDyn::write_restart(
    Core::IO::DiscretizationWriter& iowriter, const bool& forced_writerestart) const
{
  // nothing to do
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::BrownianDyn::read_restart(Core::IO::DiscretizationReader& ioreader)
{
  // nothing to do
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::BrownianDyn::run_post_compute_x(
    const Core::LinAlg::Vector<double>& xold, const Core::LinAlg::Vector<double>& dir,
    const Core::LinAlg::Vector<double>& xnew)
{
  // empty
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::BrownianDyn::update_step_state(const double& timefac_n)
{
  check_init_setup();
  // -------------------------------------------------------------------------
  // add brownian force contributions to the old structural
  // residual state vector
  // -------------------------------------------------------------------------
  std::shared_ptr<Core::LinAlg::Vector<double>>& fstructold_ptr =
      global_state().get_fstructure_old();
  fstructold_ptr->update(timefac_n, *f_brown_np_ptr_, 1.0);
  fstructold_ptr->update(-timefac_n, *f_ext_np_ptr_, 1.0);

  return;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::BrownianDyn::update_step_element()
{
  // -------------------------------------------------------------------------
  // check if timestep changes according to action dt in input file
  // -------------------------------------------------------------------------
  // todo: this needs to go somewhere else, to a more global/general place
  // (console output at this point is also very unflattering)
  //  sm_manager_ptr_->UpdateTimeAndStepSize((*GStatePtr()->get_delta_time())[0],
  //                                           GStatePtr()->get_time_n());

  return;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::BrownianDyn::determine_stress_strain()
{
  // nothing to do
  return;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::BrownianDyn::determine_energy()
{
  // nothing to do
  return;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Map>
Solid::ModelEvaluator::BrownianDyn::get_block_dof_row_map_ptr() const
{
  check_init_setup();
  return global_state().dof_row_map();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Vector<double>>
Solid::ModelEvaluator::BrownianDyn::get_current_solution_ptr() const
{
  // there are no model specific solution entries
  return nullptr;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Vector<double>>
Solid::ModelEvaluator::BrownianDyn::get_last_time_step_solution_ptr() const
{
  // there are no model specific solution entries
  return nullptr;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::BrownianDyn::post_output()
{
  check_init_setup();
  // -------------------------------------------------------------------------
  // Generate new random forces
  // -------------------------------------------------------------------------
  generate_gaussian_random_numbers();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::BrownianDyn::reset_step_state()
{
  check_init_setup();

  if (global_state().get_my_rank() == 0)
    std::cout << " NOTE: stochastic forces stay unchanged in case of DIVERCONT" << std::endl;
  /*
    // -------------------------------------------------------------------------
    // Generate new random forces
    // -------------------------------------------------------------------------
    generate_gaussian_random_numbers();
    // -------------------------------------------------------------------------
    // Update number of unconverged steps
    // -------------------------------------------------------------------------
  */
  //  sm_manager_ptr_->UpdateNumberOfUnconvergedSteps();

  /* special part in brownian for predictor: initialize disn_ and veln_ with zero;
   * this is necessary only for the following case: Assume that an iteration
   * step did not converge and is repeated with new random numbers; if the
   * failure of convergence lead to disn_ = NaN and veln_ = NaN this would affect
   * also the next trial as e.g. disn_->Update(1.0,*((*dis_)(0)),0.0); would set
   * disn_ to NaN as even 0*NaN = NaN!; this would defeat the purpose of the
   * repeated iterations with new random numbers and has thus to be avoided;
   * therefore we initialized disn_ and veln_ with zero which has no effect
   * in any other case*/
  // todo: is this the right place for this (originally done in brownian predictor,
  // should work as prediction is the next thing that is done)
  global_state_ptr()->get_dis_np()->put_scalar(0.0);
  global_state_ptr()->get_vel_np()->put_scalar(0.0);
  // we only need this in case we use Lie Group gen alpha and calculate a consistent
  // mass matrix and acc vector (i.e. we are not neglecting inertia forces)
  global_state_ptr()->get_acc_np()->put_scalar(0.0);

  global_state_ptr()->get_dis_np()->update(1.0, (*global_state_ptr()->get_dis_n()), 0.0);
  global_state_ptr()->get_vel_np()->update(1.0, (*global_state_ptr()->get_vel_n()), 0.0);
  global_state_ptr()->get_acc_np()->update(1.0, (*global_state_ptr()->get_acc_n()), 0.0);


  return;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::BrownianDyn::random_numbers_per_element()
{
  check_init();
  // -------------------------------------------------------------------------
  // maximal number of random numbers to be generated per time step for
  // any column map element of this processor
  // -------------------------------------------------------------------------
  int randomnumbersperlocalelement = 0;
  // -------------------------------------------------------------------------
  // check maximal number of nodes of an element with stochastic forces
  // on this processor
  // -------------------------------------------------------------------------
  // see whether current element needs more random numbers per time step
  // than any other before
  for (int i = 0; i < discret_ptr_->num_my_col_elements(); ++i)
  {
    Discret::Elements::Beam3Base* beamele =
        dynamic_cast<Discret::Elements::Beam3Base*>(discret_ptr_->l_col_element(i));
    if (beamele != nullptr)
    {
      randomnumbersperlocalelement =
          std::max(randomnumbersperlocalelement, beamele->how_many_random_numbers_i_need());
    }
    else if (dynamic_cast<Discret::Elements::Rigidsphere*>(discret_ptr_->l_col_element(i)) !=
             nullptr)
    {
      randomnumbersperlocalelement = std::max(randomnumbersperlocalelement,
          dynamic_cast<Discret::Elements::Rigidsphere*>(discret_ptr_->l_col_element(i))
              ->how_many_random_numbers_i_need());
    }
    else
    {
      FOUR_C_THROW("Brownian dynamics simulation not (yet) implemented for this element type.");
    }
  }
  // -------------------------------------------------------------------------
  // so far the maximal number of random numbers required per element
  // has been checked only locally on this processor; now we compare the
  // results of each processor and store the maximal one in
  // maxrandnumelement_
  // -------------------------------------------------------------------------
  maxrandnumelement_ =
      Core::Communication::max_all(randomnumbersperlocalelement, discret_ptr()->get_comm());
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::BrownianDyn::generate_gaussian_random_numbers()
{
  check_init();

  // only update random numbers and therefore stochastic forces each stochastic time step
  // note: in case of a restart, first stochastic time step can be smaller than
  // brown_dyn_state_data_.browndyn_dt, this is intended
  int browndyn_step =
      static_cast<int>((global_state().get_time_np() - global_state_ptr()->get_delta_time()[0]) /
                           brown_dyn_state_data_.browndyn_dt +
                       1.0e-8);

  if (browndyn_step == brown_dyn_state_data_.browndyn_step)
    return;
  else
    brown_dyn_state_data_.browndyn_step = browndyn_step;

  // initialize mean value 0 and standard deviation (2KT / dt)^0.5
  double meanvalue = 0.0;

  // generate gaussian random numbers for parallel use with mean value 0 and
  // standard deviation (2KT / dt)^0.5
  double standarddeviation =
      pow(2.0 * eval_browniandyn_ptr_->kt() / brown_dyn_state_data_.browndyn_dt, 0.5);

  Global::Problem::instance()->random()->set_rand_range(0.0, 1.0);

  // multivector for stochastic forces evaluated by each element based on row map
  std::shared_ptr<Core::LinAlg::MultiVector<double>> randomnumbersrow =
      eval_browniandyn_ptr_->get_random_forces();

  int numele = randomnumbersrow->local_length();
  int numperele = randomnumbersrow->num_vectors();
  int count = numele * numperele;

  // Start out with zeros
  std::vector<double> randvec(count);
  // Only generate random numbers if the distribution is not a point distribution
  if (standarddeviation > 0.0)
  {
    Global::Problem::instance()->random()->set_mean_stddev(meanvalue, standarddeviation);
    Global::Problem::instance()->random()->normal(randvec, count);
  }

  // MAXRANDFORCE is a multiple of the standard deviation
  double maxrandforcefac = eval_browniandyn_ptr_->max_rand_force();
  if (maxrandforcefac == -1.0)
  {
    for (int i = 0; i < numele; ++i)
      for (int j = 0; j < numperele; ++j)
      {
        randomnumbersrow->get_vector(j).get_values()[i] = randvec[i * numperele + j];
      }
  }
  else
  {
    for (int i = 0; i < numele; ++i)
      for (int j = 0; j < numperele; ++j)
      {
        randomnumbersrow->get_vector(j).get_values()[i] = randvec[i * numperele + j];

        if (randomnumbersrow->get_vector(j).get_values()[i] >
            maxrandforcefac * standarddeviation + meanvalue)
        {
          std::cout << "warning: stochastic force restricted according to MAXRANDFORCE"
                       " this should not happen to often"
                    << std::endl;
          randomnumbersrow->get_vector(j).get_values()[i] =
              maxrandforcefac * standarddeviation + meanvalue;
        }
        else if (randomnumbersrow->get_vector(j).local_values_as_span()[i] <
                 -maxrandforcefac * standarddeviation + meanvalue)
        {
          std::cout << "warning: stochastic force restricted according to MAXRANDFORCE"
                       " this should not happen to often"
                    << std::endl;
          randomnumbersrow->get_vector(j).get_values()[i] =
              -maxrandforcefac * standarddeviation + meanvalue;
        }
      }
  }
}


/*
----------------------------------------------------------------------------*
 | seed all random generators of this object with fixed seed if given and     |
 | with system time otherwise; seedparameter is used only in the first        |
 | case to calculate the actual seed variable based on some given fixed       |
 | seed value; note that seedparameter may be any integer, but has to be      |
 | been set in a deterministic way so that it for a certain call of this      |
 | method at a certain point in the program always the same number            |
 | whenever the program is used                                               |
 *----------------------------------------------------------------------------
void Solid::ModelEvaluator::BrownianDyn::SeedRandomGenerator()
{
  check_init();

  const int    stepn  = GStatePtr()->get_step_n() + 1;
  const double timenp = GStatePtr()->get_time_np();
  const double dt     = (*GStatePtr()->get_delta_time())[0];
  const int myrank = global_state().get_my_rank();

  // -----------------------------------------------------------------------
  // Decide if random numbers should change in every time step...
  // read time interval within the random numbers remain constant (-1.0 means no
  // prescribed time interval). This means new random numbers every
  // randnumtimeinc seconds.
  // -----------------------------------------------------------------------
  if( rand_data_.time_interv_with_const_rn == -1.0 )
  {
    // new random numbers every time step (same in each program start though)
    if ( rand_data_.randseed >= 0 )
      rand_data_.seedvariable = ( rand_data_.randseed + stepn ) * ( myrank + 1 );
    // else
    // set seed according to system time and different for each processor
    // once in the beginning (done in read_parameter globalproblem.cpp)
    // in this case we have different random numbers in each program start
    // and time step
  }
  //...or only every time_interv_with_const_rn seconds
  else
  {
    // this variable changes every time_interv_with_const_rn seconds
    int seed_differs_every_time_int =
        static_cast<int>( (timenp - dt ) / rand_data_.time_interv_with_const_rn + 1.0e-8 );

    // same each program start
    if ( rand_data_.randseed >= 0 )
    {
      rand_data_.seedvariable =
          ( rand_data_.randseed + seed_differs_every_time_int ) * ( myrank + 1 );
    }
    // .. different each program start
    else if( seed_differs_every_time_int != rand_data_.seed_differs_every_time_int )
    {
      rand_data_.seedvariable = static_cast<int>( time(nullptr) ) + 27 * ( myrank + 1 );
      rand_data_.seed_differs_every_time_int = seed_differs_every_time_int;
    }
  }
  // -----------------------------------------------------------------------
  // seed random number generator and set uni range
  // -----------------------------------------------------------------------
  Global::Problem::instance()->Random()->SetRandSeed( static_cast<unsigned int>(
rand_data_.seedvariable ) ); Global::Problem::instance()->Random()->SetRandRange( 0.0, 1.0);

}*/

FOUR_C_NAMESPACE_CLOSE
