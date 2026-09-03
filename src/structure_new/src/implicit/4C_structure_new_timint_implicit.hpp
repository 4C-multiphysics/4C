// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_STRUCTURE_NEW_TIMINT_IMPLICIT_HPP
#define FOUR_C_STRUCTURE_NEW_TIMINT_IMPLICIT_HPP

#include "4C_config.hpp"

#include "4C_adapter_str_structure.hpp"
#include "4C_structure_new_input.hpp"
#include "4C_structure_new_nln_solver_generic.hpp"
#include "4C_structure_new_timint_implicitbase.hpp"  // base class

#include <cstddef>
#include <deque>

FOUR_C_NAMESPACE_OPEN

// forward declarations ...
namespace Solid
{
  namespace IMPLICIT
  {
    class Generic;
  }  // namespace IMPLICIT
  namespace Predict
  {
    class Generic;
  }  // namespace Predict
  namespace Nln::SOLVER
  {
    class Generic;
    namespace INTERFACE
    {
      class Required;
    }  // namespace INTERFACE
  }  // namespace Nln::SOLVER

  namespace TimeInt
  {
    /** \brief Implicit time integration strategy
     *
     * */
    class Implicit : public ImplicitBase
    {
     public:
      void setup() override;

      void integrate() override;

      void integrate_step() override;

      /// set the state of the nox group and the global state data container
      /// see class \ref Adapter::StructureNew for a detailed documentation.
      void set_state(const std::shared_ptr<Core::LinAlg::Vector<double>>& x) override;

      /*! \brief nonlinear solve
       *
       *  Do the nonlinear solve, i.e. (multiple) corrector,
       *  for the time step. All boundary conditions have
       *  been set. */
      [[nodiscard]] Solid::StepStatus solve() override;

      /** \brief Identify residual
       *
       *  This method does not predict the target solution but
       *  evaluates the residual and the stiffness matrix.
       *  In partitioned solution schemes, it is better to keep the current
       *  solution instead of evaluating the initial guess (as the predictor)
       *  does. */
      void prepare_partition_step() override;

      //! Prepare time step
      void prepare_time_step() override;

      //! @name Accessors
      //! @{
      //! return the predictor
      [[nodiscard]] const Solid::Predict::Generic& predictor() const
      {
        check_init_setup();
        return *predictor_ptr_;
      }

      //! @}
      [[nodiscard]] std::shared_ptr<const Solid::Nln::SOLVER::Generic> get_nln_solver_ptr() const
      {
        return nlnsolver_ptr_;
      };

      Solid::StepAction perform_error_action(Solid::StepStatus solve_status) override;

      void finalize_successful_step() override;

      //! returns pointer to generic implicit object
      std::shared_ptr<Solid::IMPLICIT::Generic> impl_int_ptr()
      {
        check_init_setup();
        return implint_ptr_;
      };

      /// Update State Incrementally for coupled problems with monolithic approach
      void update_state_incrementally(
          std::shared_ptr<const Core::LinAlg::Vector<double>> disiterinc) override;

      void determine_stress_strain() override;

      ///  Evaluate routine for coupled problems with monolithic approach
      void evaluate(std::shared_ptr<const Core::LinAlg::Vector<double>> disiterinc) override;
      void evaluate() override;

      /** \brief Print structural jacobian matrix into a text file for later use
       *  in MATLAB
       *
       *  This routine can be activated via the input parameter
       *  %STRUCT_JACOBIAN_MATLAB. See the corresponding inpar section for more
       *  details.
       *
       *  The text file can be found in the user-provided output folder using the
       *  following file name extension:
       *
       *  [OUTPUT-FOLDER]/[OUTPUT FILE NAME]_str_jacobian_step-[STEP]_nlniter-[NEWTON-ITERATION].mtl
       *
       *  */
      void print_jacobian_in_matlab_format(const NOX::Nln::Group& curr_grp) const;

     protected:
      //! returns the current solution group
      [[nodiscard]] const ::NOX::Abstract::Group& get_solution_group() const override;

      void remap_solver_after_redistribution() override
      {
        nlnsolver_ptr_->refresh_after_redistribution();
      }

      Solid::IMPLICIT::Generic& impl_int()
      {
        check_init_setup();
        return *implint_ptr_;
      };

      Solid::Predict::Generic& predictor()
      {
        check_init_setup();
        return *predictor_ptr_;
      };

      std::shared_ptr<Solid::Predict::Generic> predictor_ptr()
      {
        check_init_setup();
        return predictor_ptr_;
      };

      [[nodiscard]] const Solid::Nln::SOLVER::Generic& nln_solver() const
      {
        check_init_setup();
        return *nlnsolver_ptr_;
      };

      Solid::Nln::SOLVER::Generic& nln_solver()
      {
        check_init_setup();
        return *nlnsolver_ptr_;
      };

      std::shared_ptr<Solid::Nln::SOLVER::Generic> nln_solver_ptr()
      {
        check_init_setup();
        return nlnsolver_ptr_;
      };

      //! @name Attribute access functions
      //@{

      //! Provide Name
      Solid::DynamicType method_name() const override;

      //! Provide number of steps, e.g. a single-step method returns 1,
      //! a \f$m\f$-multistep method returns \f$m\f$
      int method_steps() const override;

      //! Give local order of accuracy of displacement part
      int method_order_of_accuracy_dis() const override;

      //! Give local order of accuracy of velocity part
      int method_order_of_accuracy_vel() const override;

      //! Return linear error coefficient of displacements
      double method_lin_err_coeff_dis() const override;

      //! Return linear error coefficient of velocities
      double method_lin_err_coeff_vel() const override;

      ///@}

     private:
      //! Reduce the time-step size and reset state so the current step can be retried.
      void prepare_retry_with_reduced_time_step();

      //! Update time-step control history and set the time-step size for the next step.
      void time_step_control_after_successful_step();

      //! Apply a changed time-step size and refresh all dependent time-integration state.
      void set_new_time_step_size(double new_dt);

      //! ptr to the implicit time integrator object
      std::shared_ptr<Solid::IMPLICIT::Generic> implint_ptr_;

      //! ptr to the predictor object
      std::shared_ptr<Solid::Predict::Generic> predictor_ptr_;

      //! ptr to the non-linear solver object
      std::shared_ptr<Solid::Nln::SOLVER::Generic> nlnsolver_ptr_;

      //! ptr to the nox group object
      std::shared_ptr<::NOX::Abstract::Group> grp_ptr_;

      //! Number of accepted steps since the current time-step size was selected.
      size_t num_steps_at_current_dt_ = 0;

      //! Nonlinear iteration counts from the recent accepted steps used for time-step recovery.
      std::deque<size_t> number_of_nonlinear_iterations_per_step_;
    };
  }  // namespace TimeInt
}  // namespace Solid

FOUR_C_NAMESPACE_CLOSE

#endif
