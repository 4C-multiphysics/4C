// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_structure_new_model_evaluator_generic.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_solver_nonlin_nox_floating_point_exception.hpp"
#include "4C_structure_new_model_evaluator_data.hpp"
#include "4C_structure_new_timint_base.hpp"
#include "4C_structure_new_utils.hpp"
#include "4C_utils_exceptions.hpp"


FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Solid::ModelEvaluator::Generic::Generic()
    : isinit_(false),
      issetup_(false),
      eval_data_ptr_(nullptr),
      gstate_ptr_(nullptr),
      gio_ptr_(nullptr),
      discret_ptr_(nullptr),
      timint_ptr_(nullptr),
      dof_offset_(0)
{
  // empty constructor
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::Generic::init(
    const std::shared_ptr<Solid::ModelEvaluator::Data>& eval_data_ptr,
    const std::shared_ptr<Solid::TimeInt::BaseDataGlobalState>& gstate_ptr,
    const std::shared_ptr<Solid::TimeInt::BaseDataIO>& gio_ptr,
    const std::shared_ptr<Solid::Integrator>& int_ptr,
    const std::shared_ptr<const Solid::TimeInt::Base>& timint_ptr, const int& dof_offset)
{
  // call setup after init()
  issetup_ = false;

  eval_data_ptr_ = eval_data_ptr;
  gstate_ptr_ = gstate_ptr;
  gio_ptr_ = gio_ptr;
  discret_ptr_ = gstate_ptr->get_discret();
  int_ptr_ = int_ptr;
  timint_ptr_ = timint_ptr;
  dof_offset_ = dof_offset;

  isinit_ = true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::Generic::check_init_setup() const
{
  FOUR_C_ASSERT(is_init() and is_setup(), "Call init() and setup() first!");
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::Generic::check_init() const
{
  FOUR_C_ASSERT(is_init(), "Call init() first!");
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Solid::ModelEvaluator::Data& Solid::ModelEvaluator::Generic::eval_data()
{
  check_init();
  return *eval_data_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const Solid::ModelEvaluator::Data& Solid::ModelEvaluator::Generic::eval_data() const
{
  check_init();
  return *eval_data_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<Solid::ModelEvaluator::Data>& Solid::ModelEvaluator::Generic::eval_data_ptr()
{
  check_init();
  return eval_data_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Solid::TimeInt::BaseDataGlobalState& Solid::ModelEvaluator::Generic::global_state()
{
  check_init();
  return *gstate_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<Solid::TimeInt::BaseDataGlobalState>&
Solid::ModelEvaluator::Generic::global_state_ptr()
{
  check_init();
  return gstate_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const Solid::TimeInt::BaseDataGlobalState& Solid::ModelEvaluator::Generic::global_state() const
{
  check_init();
  return *gstate_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Solid::TimeInt::BaseDataIO& Solid::ModelEvaluator::Generic::global_in_output()
{
  check_init();
  return *gio_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<Solid::TimeInt::BaseDataIO> Solid::ModelEvaluator::Generic::global_in_output_ptr()
{
  check_init();
  return gio_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const Solid::TimeInt::BaseDataIO& Solid::ModelEvaluator::Generic::global_in_output() const
{
  check_init();
  return *gio_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Core::FE::Discretization& Solid::ModelEvaluator::Generic::discret()
{
  check_init();
  return *discret_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<Core::FE::Discretization>& Solid::ModelEvaluator::Generic::discret_ptr()
{
  check_init();
  return discret_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const Core::FE::Discretization& Solid::ModelEvaluator::Generic::discret() const
{
  check_init();
  return *discret_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Solid::Integrator& Solid::ModelEvaluator::Generic::integrator()
{
  check_init();
  return *int_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const Solid::Integrator& Solid::ModelEvaluator::Generic::integrator() const
{
  check_init();
  return *int_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<Solid::Integrator>& Solid::ModelEvaluator::Generic::integrator_ptr()
{
  return int_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const Solid::TimeInt::Base& Solid::ModelEvaluator::Generic::tim_int() const
{
  check_init();
  return *timint_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const int& Solid::ModelEvaluator::Generic::dof_offset() const
{
  check_init();
  return dof_offset_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> Solid::ModelEvaluator::Generic::get_fext_incr() const
{
  check_init_setup();
  const Core::LinAlg::Vector<double>& fextn = *global_state().get_fext_n();
  const Core::LinAlg::Vector<double>& fextnp = *global_state().get_fext_np();

  std::shared_ptr<Core::LinAlg::Vector<double>> fext_incr =
      std::make_shared<Core::LinAlg::Vector<double>>(fextnp);
  fext_incr->update(-1.0, fextn, 1.0);

  return fext_incr;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Solid::ModelEvaluator::Generic::eval_error_check() const
{
  // --- Did an exception occur during the evaluation process? -----------------
  bool ok = true;
  int fp_err = NOX::Nln::FloatingPointException::check_and_print(std::cout);
  if (fp_err)
  {
    ok = false;
    std::cout << "FLOATING POINT EXCEPTION occurred on proc "
                 "#"
              << Core::Communication::my_mpi_rank(gstate_ptr_->get_comm()) << ".\n";
  }

  // --- Did the element evaluation detect an error? ---------------------------
  ok = (ok and (not eval_data_ptr_->is_ele_eval_error()));

  if (eval_data_ptr_->is_ele_eval_error())
    std::cout << "ELEMENT EVALUATION failed on proc "
                 "#"
              << Core::Communication::my_mpi_rank(gstate_ptr_->get_comm()) << ".\n"
              << "(Error: "
              << Solid::Elements::eval_error_flag_to_string(
                     eval_data_ptr_->get_ele_eval_error_flag())
              << ")\n";

  // reset the flag
  eval_data_ptr_->set_ele_eval_error_flag(Solid::Elements::ele_error_none);

  // --- check for local errors on each proc and communicate the information ---
  int lerr = (ok ? 0 : 1);
  int gerr = 0;
  gerr = Core::Communication::sum_all(lerr, gstate_ptr_->get_comm());
  return (gerr == 0);
}

FOUR_C_NAMESPACE_CLOSE
