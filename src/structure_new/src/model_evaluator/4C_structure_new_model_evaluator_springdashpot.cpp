/*-----------------------------------------------------------*/
/*! \file

\brief Evaluation and assembly of all spring dashpot terms


\level 3

*/
/*-----------------------------------------------------------*/

#include "4C_structure_new_model_evaluator_springdashpot.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_structure.hpp"
#include "4C_io.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_sparseoperator.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_structure_new_model_evaluator_data.hpp"
#include "4C_structure_new_timint_base.hpp"
#include "4C_structure_new_utils.hpp"
#include "4C_utils_exceptions.hpp"

#include <Teuchos_ParameterList.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Solid::ModelEvaluator::SpringDashpot::SpringDashpot()
    : disnp_ptr_(Teuchos::null),
      velnp_ptr_(Teuchos::null),
      stiff_spring_ptr_(Teuchos::null),
      fspring_np_ptr_(Teuchos::null)
{
  // empty
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Solid::ModelEvaluator::SpringDashpot::setup()
{
  FOUR_C_ASSERT(is_init(), "init() has not been called, yet!");

  // get all spring dashpot conditions
  std::vector<Teuchos::RCP<Core::Conditions::Condition>> springdashpots;
  discret().get_condition("RobinSpringDashpot", springdashpots);

  // new instance of spring dashpot BC for each condition
  for (auto& springdashpot : springdashpots)
    springs_.emplace_back(
        Teuchos::rcp(new CONSTRAINTS::SpringDashpot(discret_ptr(), springdashpot)));

  // setup the displacement pointer
  disnp_ptr_ = global_state().get_dis_np();
  velnp_ptr_ = global_state().get_vel_np();

  fspring_np_ptr_ =
      Teuchos::rcp(new Core::LinAlg::Vector<double>(*global_state().dof_row_map_view()));
  stiff_spring_ptr_ = Teuchos::rcp(
      new Core::LinAlg::SparseMatrix(*global_state().dof_row_map_view(), 81, true, true));

  // set flag
  issetup_ = true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Solid::ModelEvaluator::SpringDashpot::reset(const Core::LinAlg::Vector<double>& x)
{
  check_init_setup();

  // loop over all spring dashpot conditions and reset them
  for (const auto& spring : springs_) spring->reset_newton();

  // update the structural displacement vector
  disnp_ptr_ = global_state().get_dis_np();

  // update the structural displacement vector
  velnp_ptr_ = global_state().get_vel_np();

  fspring_np_ptr_->PutScalar(0.0);
  stiff_spring_ptr_->zero();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool Solid::ModelEvaluator::SpringDashpot::evaluate_force()
{
  check_init_setup();

  Teuchos::ParameterList springdashpotparams;
  // loop over all spring dashpot conditions and evaluate them
  fspring_np_ptr_ =
      Teuchos::rcp(new Core::LinAlg::Vector<double>(*global_state().dof_row_map_view()));
  for (const auto& spring : springs_)
  {
    const CONSTRAINTS::SpringDashpot::SpringType stype = spring->get_spring_type();

    if (stype == CONSTRAINTS::SpringDashpot::xyz or
        stype == CONSTRAINTS::SpringDashpot::refsurfnormal)
    {
      springdashpotparams.set("total time", global_state().get_time_np());
      spring->evaluate_robin(
          Teuchos::null, fspring_np_ptr_, disnp_ptr_, velnp_ptr_, springdashpotparams);
    }
    if (stype == CONSTRAINTS::SpringDashpot::cursurfnormal)
    {
      springdashpotparams.set("dt", (*global_state().get_delta_time())[0]);
      spring->evaluate_force(*fspring_np_ptr_, disnp_ptr_, velnp_ptr_, springdashpotparams);
    }
  }

  return true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool Solid::ModelEvaluator::SpringDashpot::evaluate_stiff()
{
  check_init_setup();

  fspring_np_ptr_ =
      Teuchos::rcp(new Core::LinAlg::Vector<double>(*global_state().dof_row_map_view(), true));

  // factors from time-integrator for derivative of d(v_{n+1}) / d(d_{n+1})
  // needed for stiffness contribution from dashpot
  const double fac_vel = eval_data().get_tim_int_factor_vel();
  const double fac_disp = eval_data().get_tim_int_factor_disp();
  const double time_fac = fac_vel / fac_disp;
  Teuchos::ParameterList springdashpotparams;
  if (fac_vel > 0.0) springdashpotparams.set("time_fac", time_fac);

  // loop over all spring dashpot conditions and evaluate them
  for (const auto& spring : springs_)
  {
    const CONSTRAINTS::SpringDashpot::SpringType stype = spring->get_spring_type();

    if (stype == CONSTRAINTS::SpringDashpot::xyz or
        stype == CONSTRAINTS::SpringDashpot::refsurfnormal)
    {
      springdashpotparams.set("total time", global_state().get_time_np());
      spring->evaluate_robin(
          stiff_spring_ptr_, Teuchos::null, disnp_ptr_, velnp_ptr_, springdashpotparams);
    }
    if (stype == CONSTRAINTS::SpringDashpot::cursurfnormal)
    {
      springdashpotparams.set("dt", (*global_state().get_delta_time())[0]);
      spring->evaluate_force_stiff(
          *stiff_spring_ptr_, *fspring_np_ptr_, disnp_ptr_, velnp_ptr_, springdashpotparams);
    }
  }

  if (not stiff_spring_ptr_->filled()) stiff_spring_ptr_->complete();

  return true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool Solid::ModelEvaluator::SpringDashpot::evaluate_force_stiff()
{
  check_init_setup();

  // get displacement DOFs
  fspring_np_ptr_ =
      Teuchos::rcp(new Core::LinAlg::Vector<double>(*global_state().dof_row_map(), true));

  // factors from time-integrator for derivative of d(v_{n+1}) / d(d_{n+1})
  // needed for stiffness contribution from dashpot
  const double fac_vel = eval_data().get_tim_int_factor_vel();
  const double fac_disp = eval_data().get_tim_int_factor_disp();
  const double time_fac = fac_vel / fac_disp;
  Teuchos::ParameterList springdashpotparams;

  if (fac_vel > 0.0) springdashpotparams.set("time_fac", time_fac);

  // loop over all spring dashpot conditions and evaluate them
  for (const auto& spring : springs_)
  {
    const CONSTRAINTS::SpringDashpot::SpringType stype = spring->get_spring_type();

    if (stype == CONSTRAINTS::SpringDashpot::xyz or
        stype == CONSTRAINTS::SpringDashpot::refsurfnormal)
    {
      springdashpotparams.set("total time", global_state().get_time_np());
      spring->evaluate_robin(
          stiff_spring_ptr_, fspring_np_ptr_, disnp_ptr_, velnp_ptr_, springdashpotparams);
    }
    if (stype == CONSTRAINTS::SpringDashpot::cursurfnormal)
    {
      springdashpotparams.set("dt", (*global_state().get_delta_time())[0]);
      spring->evaluate_force_stiff(
          *stiff_spring_ptr_, *fspring_np_ptr_, disnp_ptr_, velnp_ptr_, springdashpotparams);
    }
  }

  if (not stiff_spring_ptr_->filled()) stiff_spring_ptr_->complete();

  return true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool Solid::ModelEvaluator::SpringDashpot::assemble_force(
    Core::LinAlg::Vector<double>& f, const double& timefac_np) const
{
  Core::LinAlg::assemble_my_vector(1.0, f, timefac_np, *fspring_np_ptr_);
  return true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool Solid::ModelEvaluator::SpringDashpot::assemble_jacobian(
    Core::LinAlg::SparseOperator& jac, const double& timefac_np) const
{
  Teuchos::RCP<Core::LinAlg::SparseMatrix> jac_dd_ptr = global_state().extract_displ_block(jac);
  jac_dd_ptr->add(*stiff_spring_ptr_, false, timefac_np, 1.0);
  // no need to keep it
  stiff_spring_ptr_->zero();
  // nothing to do
  return true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Solid::ModelEvaluator::SpringDashpot::write_restart(
    Core::IO::DiscretizationWriter& iowriter, const bool& forced_writerestart) const
{
  // row maps for export
  Teuchos::RCP<Core::LinAlg::Vector<double>> springoffsetprestr =
      Teuchos::rcp(new Core::LinAlg::Vector<double>(*discret().dof_row_map()));
  Teuchos::RCP<Epetra_MultiVector> springoffsetprestr_old =
      Teuchos::rcp(new Epetra_MultiVector(*(discret().node_row_map()), 3, true));

  // collect outputs from all spring dashpot conditions
  for (const auto& spring : springs_)
  {
    // get spring type from current condition
    const CONSTRAINTS::SpringDashpot::SpringType stype = spring->get_spring_type();

    if (stype == CONSTRAINTS::SpringDashpot::xyz or
        stype == CONSTRAINTS::SpringDashpot::refsurfnormal)
      spring->output_prestr_offset(springoffsetprestr);
    if (stype == CONSTRAINTS::SpringDashpot::cursurfnormal)
      spring->output_prestr_offset_old(springoffsetprestr_old);
  }

  // write vector to output for restart
  iowriter.write_vector("springoffsetprestr", springoffsetprestr);
  // write vector to output for restart
  iowriter.write_multi_vector("springoffsetprestr_old", springoffsetprestr_old);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Solid::ModelEvaluator::SpringDashpot::read_restart(Core::IO::DiscretizationReader& ioreader)
{
  Teuchos::RCP<Core::LinAlg::Vector<double>> tempvec =
      Teuchos::rcp(new Core::LinAlg::Vector<double>(*discret().dof_row_map()));
  Teuchos::RCP<Epetra_MultiVector> tempvecold =
      Teuchos::rcp(new Epetra_MultiVector(*(discret().node_row_map()), 3, true));

  ioreader.read_vector(tempvec, "springoffsetprestr");
  ioreader.read_multi_vector(tempvecold, "springoffsetprestr_old");

  // loop over all spring dashpot conditions and set restart
  for (const auto& spring : springs_)
  {
    // get spring type from current condition
    const CONSTRAINTS::SpringDashpot::SpringType stype = spring->get_spring_type();

    if (stype == CONSTRAINTS::SpringDashpot::xyz or
        stype == CONSTRAINTS::SpringDashpot::refsurfnormal)
      spring->set_restart(tempvec);
    if (stype == CONSTRAINTS::SpringDashpot::cursurfnormal) spring->set_restart_old(tempvecold);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Solid::ModelEvaluator::SpringDashpot::update_step_state(const double& timefac_n)
{
  // add the old time factor scaled contributions to the residual
  Teuchos::RCP<Core::LinAlg::Vector<double>>& fstructold_ptr = global_state().get_fstructure_old();
  fstructold_ptr->Update(timefac_n, *fspring_np_ptr_, 1.0);

  // check for prestressing and reset if necessary
  const Inpar::Solid::PreStress prestress_type = tim_int().get_data_sdyn().get_pre_stress_type();
  const double prestress_time = tim_int().get_data_sdyn().get_pre_stress_time();

  if (prestress_type != Inpar::Solid::PreStress::none &&
      global_state().get_time_np() <= prestress_time + 1.0e-15)
  {
    switch (prestress_type)
    {
      case Inpar::Solid::PreStress::mulf:
      case Inpar::Solid::PreStress::material_iterative:
        for (const auto& spring : springs_) spring->reset_prestress(global_state().get_dis_np());
      default:
        break;
    }
  }
  for (const auto& spring : springs_) spring->update();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Solid::ModelEvaluator::SpringDashpot::output_step_state(
    Core::IO::DiscretizationWriter& iowriter) const
{
  // row maps for export
  Teuchos::RCP<Core::LinAlg::Vector<double>> gap =
      Teuchos::rcp(new Core::LinAlg::Vector<double>(*(discret().node_row_map()), true));
  Teuchos::RCP<Epetra_MultiVector> normals =
      Teuchos::rcp(new Epetra_MultiVector(*(discret().node_row_map()), 3, true));
  Teuchos::RCP<Epetra_MultiVector> springstress =
      Teuchos::rcp(new Epetra_MultiVector(*(discret().node_row_map()), 3, true));

  // collect outputs from all spring dashpot conditions
  bool found_cursurfnormal = false;
  for (const auto& spring : springs_)
  {
    spring->output_gap_normal(gap, normals, springstress);

    // get spring type from current condition
    const CONSTRAINTS::SpringDashpot::SpringType stype = spring->get_spring_type();
    if (stype == CONSTRAINTS::SpringDashpot::cursurfnormal) found_cursurfnormal = true;
  }

  // write vectors to output
  if (found_cursurfnormal)
  {
    iowriter.write_vector("gap", gap);
    iowriter.write_multi_vector("curnormals", normals);
  }

  // write spring stress if defined in io-flag
  if (Global::Problem::instance()->io_params().get<bool>("OUTPUT_SPRING"))
    iowriter.write_multi_vector("springstress", springstress);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Solid::ModelEvaluator::SpringDashpot::reset_step_state()
{
  check_init_setup();

  for (auto& spring : springs_)
  {
    spring->reset_step_state();
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::RCP<const Epetra_Map> Solid::ModelEvaluator::SpringDashpot::get_block_dof_row_map_ptr()
    const
{
  check_init_setup();
  return global_state().dof_row_map();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::RCP<const Core::LinAlg::Vector<double>>
Solid::ModelEvaluator::SpringDashpot::get_current_solution_ptr() const
{
  // there are no model specific solution entries
  return Teuchos::null;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::RCP<const Core::LinAlg::Vector<double>>
Solid::ModelEvaluator::SpringDashpot::get_last_time_step_solution_ptr() const
{
  // there are no model specific solution entries
  return Teuchos::null;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Solid::ModelEvaluator::SpringDashpot::post_output() { check_init_setup(); }

FOUR_C_NAMESPACE_CLOSE
