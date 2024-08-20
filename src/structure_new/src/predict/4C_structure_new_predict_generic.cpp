/*-----------------------------------------------------------*/
/*! \file

\brief Generic class for all predictors.


\level 3

*/
/*-----------------------------------------------------------*/

#include "4C_structure_new_predict_generic.hpp"

#include "4C_io_pstream.hpp"
#include "4C_solver_nonlin_nox_group.hpp"
#include "4C_structure_new_dbc.hpp"
#include "4C_structure_new_impl_generic.hpp"
#include "4C_structure_new_model_evaluator_manager.hpp"
#include "4C_structure_new_timint_base.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Solid::Predict::Generic::Generic()
    : isinit_(false),
      issetup_(false),
      type_(Inpar::Solid::pred_vague),
      implint_ptr_(Teuchos::null),
      dbc_ptr_(Teuchos::null),
      noxparams_ptr_(Teuchos::null)
{
  // empty
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::Predict::Generic::init(const enum Inpar::Solid::PredEnum& type,
    const Teuchos::RCP<Solid::IMPLICIT::Generic>& implint_ptr,
    const Teuchos::RCP<Solid::Dbc>& dbc_ptr,
    const Teuchos::RCP<Solid::TimeInt::BaseDataGlobalState>& gstate_ptr,
    const Teuchos::RCP<Solid::TimeInt::BaseDataIO>& iodata_ptr,
    const Teuchos::RCP<Teuchos::ParameterList>& noxparams_ptr)
{
  issetup_ = false;

  // initialize the predictor type
  type_ = type;
  implint_ptr_ = implint_ptr;
  dbc_ptr_ = dbc_ptr;
  gstate_ptr_ = gstate_ptr;
  iodata_ptr_ = iodata_ptr;
  noxparams_ptr_ = noxparams_ptr;

  isinit_ = true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::Predict::Generic::pre_predict(::NOX::Abstract::Group& grp)
{
  check_init_setup();
  print();
  dbc_ptr_->update_loc_sys_manager();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::Predict::Generic::predict(::NOX::Abstract::Group& grp)
{
  check_init_setup();
  bool& ispredict = gstate_ptr_->is_predict();
  ispredict = true;

  // pre-process the prediction step
  pre_predict(grp);

  // compute the actual prediction step
  compute(grp);

  // post-process the prediction step
  post_predict(grp);

  ispredict = false;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::Predict::Generic::post_predict(::NOX::Abstract::Group& grp)
{
  check_init_setup();

  dbc().apply_dirichlet_bc(global_state().get_time_np(), global_state().get_dis_np(),
      global_state().get_vel_np(), global_state().get_acc_np(), false);

  // Create the new solution vector
  Teuchos::RCP<::NOX::Epetra::Vector> x_vec = global_state().create_global_vector(
      TimeInt::BaseDataGlobalState::VecInitType::init_current_state, impl_int().model_eval_ptr());
  // resets all isValid flags
  grp.setX(*x_vec);

  NOX::Nln::Group* nlngrp_ptr = dynamic_cast<NOX::Nln::Group*>(&grp);
  FOUR_C_ASSERT(nlngrp_ptr != nullptr, "Group cast failed!");
  // evaluate the right hand side and the jacobian
  implint_ptr_->set_is_predictor_state(true);
  nlngrp_ptr->compute_f_and_jacobian();
  implint_ptr_->set_is_predictor_state(false);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::string Solid::Predict::Generic::name() const
{
  check_init();
  return Inpar::Solid::pred_enum_string(type_);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::Predict::Generic::check_init() const { FOUR_C_ASSERT(is_init(), "Call init() first!"); }

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::Predict::Generic::check_init_setup() const
{
  FOUR_C_ASSERT(is_init() and is_setup(), "Call init() and setup() first!");
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<Solid::IMPLICIT::Generic>& Solid::Predict::Generic::impl_int_ptr()
{
  check_init();
  return implint_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Solid::IMPLICIT::Generic& Solid::Predict::Generic::impl_int()
{
  check_init();
  return *implint_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<Solid::Dbc>& Solid::Predict::Generic::dbc_ptr()
{
  check_init();
  return dbc_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Solid::Dbc& Solid::Predict::Generic::dbc()
{
  check_init();
  return *dbc_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<Solid::TimeInt::BaseDataGlobalState>& Solid::Predict::Generic::global_state_ptr()
{
  check_init();
  return gstate_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Solid::TimeInt::BaseDataGlobalState& Solid::Predict::Generic::global_state()
{
  check_init();
  return *gstate_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<Solid::TimeInt::BaseDataIO>& Solid::Predict::Generic::io_data_ptr()
{
  check_init();
  return iodata_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Solid::TimeInt::BaseDataIO& Solid::Predict::Generic::io_data()
{
  check_init();
  return *iodata_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const Solid::TimeInt::BaseDataGlobalState& Solid::Predict::Generic::global_state() const
{
  check_init();
  return *gstate_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<Teuchos::ParameterList>& Solid::Predict::Generic::nox_params_ptr()
{
  check_init();
  return noxparams_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::ParameterList& Solid::Predict::Generic::nox_params()
{
  check_init();
  return *noxparams_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::Predict::Generic::print() const
{
  check_init_setup();
  if (gstate_ptr_->get_my_rank() == 0 and iodata_ptr_->get_print2_screen_every_n_step() and
      gstate_ptr_->get_step_n() % iodata_ptr_->get_print2_screen_every_n_step() == 0)
  {
    Core::IO::cout << "=== Structural predictor: " << name().c_str() << " ===" << Core::IO::endl;
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Solid::Predict::Generic::pre_apply_force_external(Epetra_Vector& fextnp) const
{
  // do nothing
  return false;
}

FOUR_C_NAMESPACE_CLOSE
