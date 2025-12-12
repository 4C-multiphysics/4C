// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_structure_new_model_evaluator_lagpenconstraint.hpp"

#include "4C_constraint_lagpenconstraint_noxinterface.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_sparseoperator.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_structure_new_timint_base.hpp"
#include "4C_structure_new_utils.hpp"
#include "4C_utils_exceptions.hpp"

#include <Teuchos_ParameterList.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Solid::ModelEvaluator::LagPenConstraint::LagPenConstraint()
    : disnp_ptr_(nullptr),
      stiff_constr_ptr_(nullptr),
      fstrconstr_np_ptr_(nullptr),
      noxinterface_ptr_(nullptr),
      noxinterface_prec_ptr_(nullptr)
{
  // empty
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Solid::ModelEvaluator::LagPenConstraint::setup()
{
  check_init();

  // build the NOX::Nln::CONSTRAINT::Interface::Required object
  noxinterface_ptr_ = std::make_shared<LAGPENCONSTRAINT::NoxInterface>();
  noxinterface_ptr_->init(global_state_ptr());
  noxinterface_ptr_->setup();

  // build the NOX::Nln::CONSTRAINT::Interface::Preconditioner object
  noxinterface_prec_ptr_ = std::make_shared<LAGPENCONSTRAINT::NoxInterfacePrec>();
  noxinterface_prec_ptr_->init(global_state_ptr());
  noxinterface_prec_ptr_->setup();

  std::shared_ptr<Core::FE::Discretization> dis = discret_ptr();

  // setup the displacement pointer
  disnp_ptr_ = global_state().get_dis_np();

  // contributions of constraints to structural rhs and stiffness
  fstrconstr_np_ptr_ =
      std::make_shared<Core::LinAlg::Vector<double>>(*global_state().dof_row_map_view());
  stiff_constr_ptr_ = std::make_shared<Core::LinAlg::SparseMatrix>(
      *global_state().dof_row_map_view(), 81, true, true);

  // ToDo: we do not want to hand in the structural dynamics parameter list
  // to the manager in the future! -> get rid of it as soon as old
  // time-integration dies ...
  // initialize constraint manager
  constrman_ = std::make_shared<Constraints::ConstrManager>();
  constrman_->init(dis, Global::Problem::instance()->structural_dynamic_params());
  constrman_->setup(disnp_ptr_, Global::Problem::instance()->structural_dynamic_params());

  // set flag
  issetup_ = true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Solid::ModelEvaluator::LagPenConstraint::reset(const Core::LinAlg::Vector<double>& x)
{
  check_init_setup();

  // update the structural displacement vector
  disnp_ptr_ = global_state().get_dis_np();

  fstrconstr_np_ptr_->put_scalar(0.0);
  stiff_constr_ptr_->zero();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool Solid::ModelEvaluator::LagPenConstraint::evaluate_force()
{
  check_init_setup();

  double time_np = global_state().get_time_np();
  Teuchos::ParameterList pcon;  // empty parameter list
  std::shared_ptr<const Core::LinAlg::Vector<double>> disn = global_state().get_dis_n();

  // only forces are evaluated!
  constrman_->evaluate_force_stiff(time_np, disn, disnp_ptr_, fstrconstr_np_ptr_, nullptr, pcon);

  return true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool Solid::ModelEvaluator::LagPenConstraint::evaluate_stiff()
{
  check_init_setup();

  double time_np = global_state().get_time_np();
  Teuchos::ParameterList pcon;  // empty parameter list
  std::shared_ptr<const Core::LinAlg::Vector<double>> disn = global_state().get_dis_n();

  // only stiffnesses are evaluated!
  constrman_->evaluate_force_stiff(time_np, disn, disnp_ptr_, nullptr, stiff_constr_ptr_, pcon);

  if (not stiff_constr_ptr_->filled()) stiff_constr_ptr_->complete();

  return true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool Solid::ModelEvaluator::LagPenConstraint::evaluate_force_stiff()
{
  check_init_setup();

  double time_np = global_state().get_time_np();
  Teuchos::ParameterList pcon;  // empty parameter list
  std::shared_ptr<const Core::LinAlg::Vector<double>> disn = global_state().get_dis_n();

  constrman_->evaluate_force_stiff(
      time_np, disn, disnp_ptr_, fstrconstr_np_ptr_, stiff_constr_ptr_, pcon);

  if (not stiff_constr_ptr_->filled()) stiff_constr_ptr_->complete();

  return true;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool Solid::ModelEvaluator::LagPenConstraint::assemble_force(
    Core::LinAlg::Vector<double>& f, const double& timefac_np) const
{
  std::shared_ptr<const Core::LinAlg::Vector<double>> block_vec_ptr = nullptr;

  Core::LinAlg::assemble_my_vector(1.0, f, timefac_np, *fstrconstr_np_ptr_);

  if (noxinterface_prec_ptr_->is_saddle_point_system())
  {
    // assemble constraint rhs
    block_vec_ptr = constrman_->get_error();

    if (!block_vec_ptr)
      FOUR_C_THROW(
          "The constraint model vector is a nullptr pointer, although \n"
          "the structural part indicates, that constraint contributions \n"
          "are present!");

    Core::LinAlg::assemble_my_vector(1.0, f, timefac_np, *block_vec_ptr);
  }

  return true;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool Solid::ModelEvaluator::LagPenConstraint::assemble_jacobian(
    Core::LinAlg::SparseOperator& jac, const double& timefac_np) const
{
  std::shared_ptr<Core::LinAlg::SparseMatrix> block_ptr = nullptr;

  // --- Kdd - block ---------------------------------------------------
  std::shared_ptr<Core::LinAlg::SparseMatrix> jac_dd_ptr = global_state().extract_displ_block(jac);
  Core::LinAlg::matrix_add(*stiff_constr_ptr_, false, timefac_np, *jac_dd_ptr, 1.0);
  // no need to keep it
  stiff_constr_ptr_->zero();

  if (noxinterface_prec_ptr_->is_saddle_point_system())
  {
    // --- Kdz - block - scale with time-integrator dependent value!-----
    block_ptr =
        (std::dynamic_pointer_cast<Core::LinAlg::SparseMatrix>(constrman_->get_constr_matrix()));
    block_ptr->scale(timefac_np);
    global_state().assign_model_block(jac, *block_ptr, type(), Solid::MatBlockType::displ_lm);
    // reset the block pointer, just to be on the safe side
    block_ptr = nullptr;

    // --- Kzd - block - no scaling of this block (cf. diss Kloeppel p78)
    block_ptr = Core::LinAlg::matrix_transpose(
        *std::dynamic_pointer_cast<Core::LinAlg::SparseMatrix>(constrman_->get_constr_matrix()));
    global_state().assign_model_block(jac, *block_ptr, type(), Solid::MatBlockType::lm_displ);
    // reset the block pointer, just to be on the safe side
    block_ptr = nullptr;
  }

  return true;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Solid::ModelEvaluator::LagPenConstraint::write_restart(
    Core::IO::DiscretizationWriter& iowriter, const bool& forced_writerestart) const
{
  iowriter.write_vector("lagrmultiplier", constrman_->get_lagr_mult_vector());
  iowriter.write_vector("refconval", constrman_->get_ref_base_values());
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Solid::ModelEvaluator::LagPenConstraint::read_restart(Core::IO::DiscretizationReader& ioreader)
{
  double time_n = global_state().get_time_n();
  constrman_->read_restart(ioreader, time_n);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Solid::ModelEvaluator::LagPenConstraint::run_post_compute_x(
    const Core::LinAlg::Vector<double>& xold, const Core::LinAlg::Vector<double>& dir,
    const Core::LinAlg::Vector<double>& xnew)
{
  check_init_setup();

  Core::LinAlg::Vector<double> lagmult_incr(*get_block_dof_row_map_ptr());

  Core::LinAlg::export_to(dir, lagmult_incr);


  const auto& lm = constrman_->get_lagr_mult_vector();

  // perform update based on matching maps
  if (!lm->get_map().same_as(lagmult_incr.get_map()))
  {
    Core::LinAlg::Vector<double> lagmult_incr_on_lm(lm->get_map());
    Core::LinAlg::export_to(lagmult_incr, lagmult_incr_on_lm);
    constrman_->update_lagr_mult(lagmult_incr_on_lm);
  }
  else
  {
    constrman_->update_lagr_mult(lagmult_incr);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Solid::ModelEvaluator::LagPenConstraint::update_step_state(const double& timefac_n)
{
  constrman_->update();

  // add the constraint force contributions to the old structural
  // residual state vector
  if (fstrconstr_np_ptr_)
  {
    std::shared_ptr<Core::LinAlg::Vector<double>>& fstructold_ptr =
        global_state().get_fstructure_old();
    fstructold_ptr->update(timefac_n, *fstrconstr_np_ptr_, 1.0);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Solid::ModelEvaluator::LagPenConstraint::update_step_element()
{
  // empty
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Solid::ModelEvaluator::LagPenConstraint::determine_stress_strain()
{
  // nothing to do
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Solid::ModelEvaluator::LagPenConstraint::determine_energy()
{
  // nothing to do
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Solid::ModelEvaluator::LagPenConstraint::reset_step_state()
{
  check_init_setup();

  FOUR_C_THROW("Not yet implemented");
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
const std::shared_ptr<LAGPENCONSTRAINT::NoxInterface>&
Solid::ModelEvaluator::LagPenConstraint::nox_interface_ptr()
{
  check_init_setup();

  return noxinterface_ptr_;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
const std::shared_ptr<LAGPENCONSTRAINT::NoxInterfacePrec>&
Solid::ModelEvaluator::LagPenConstraint::nox_interface_prec_ptr()
{
  check_init_setup();

  return noxinterface_prec_ptr_;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Map>
Solid::ModelEvaluator::LagPenConstraint::get_block_dof_row_map_ptr() const
{
  check_init_setup();

  if (noxinterface_prec_ptr_->is_saddle_point_system())
  {
    return constrman_->get_constraint_map();
  }
  else
  {
    return global_state().dof_row_map();
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Vector<double>>
Solid::ModelEvaluator::LagPenConstraint::get_current_solution_ptr() const
{
  // there are no model specific solution entries
  return nullptr;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Vector<double>>
Solid::ModelEvaluator::LagPenConstraint::get_last_time_step_solution_ptr() const
{
  // there are no model specific solution entries
  return nullptr;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Solid::ModelEvaluator::LagPenConstraint::post_output()
{
  check_init_setup();
  // empty
}  // post_output()

FOUR_C_NAMESPACE_CLOSE
