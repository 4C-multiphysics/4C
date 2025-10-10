// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_STRUCTURE_NEW_MODEL_EVALUATOR_GENERIC_HPP
#define FOUR_C_STRUCTURE_NEW_MODEL_EVALUATOR_GENERIC_HPP

#include "4C_config.hpp"

#include "4C_inpar_structure.hpp"

#include <memory>

// forward declarations
class Map;
namespace NOX
{
  namespace Solver
  {
    class Generic;
  }  // namespace Solver
}  // namespace NOX
#include "4C_utils_parameter_list.fwd.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{
  template <typename T>
  class Vector;
  class Map;
}  // namespace Core::LinAlg

namespace NOX
{
  namespace Nln
  {
    class Group;
    enum class CorrectionType : int;
  }  // namespace Nln
}  // namespace NOX

namespace Core::LinAlg
{
  class SparseOperator;
  class SparseMatrix;
}  // namespace Core::LinAlg

namespace Core::IO
{
  class DiscretizationWriter;
  class DiscretizationReader;
}  // namespace Core::IO

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Solid
{
  class Integrator;

  namespace TimeInt
  {
    class BaseDataGlobalState;
    class BaseDataIO;
    class Base;
  }  // namespace TimeInt

  namespace ModelEvaluator
  {
    class Data;

    /*! \brief Abstract base class of all model evaluators
     *
     *  This class summarizes the functionality which all model evaluators share
     *  and/or have to implement. Look in the derived classes for examples. A minimal
     *  example can be found at \ref Solid::ModelEvaluator::PartitionedFSI.
     *

     *  */
    class Generic
    {
     public:
      //! constructor
      Generic();

      //! destructor
      virtual ~Generic() = default;

      /*! \brief Initialize the class variables
       *
       * \todo Complete documentation of input parameters
       *
       * @param eval_data_ptr
       * @param gstate_ptr
       * @param gio_ptr
       * @param int_ptr
       * @param timint_ptr
       * @param[in] dof_offset
       */
      virtual void init(const std::shared_ptr<Solid::ModelEvaluator::Data>& eval_data_ptr,
          const std::shared_ptr<Solid::TimeInt::BaseDataGlobalState>& gstate_ptr,
          const std::shared_ptr<Solid::TimeInt::BaseDataIO>& gio_ptr,
          const std::shared_ptr<Solid::Integrator>& int_ptr,
          const std::shared_ptr<const Solid::TimeInt::Base>& timint_ptr, const int& dof_offset);

      //! setup class variables
      virtual void setup() = 0;

     protected:
      //! Returns true, if init() has been called
      inline const bool& is_init() const { return isinit_; };

      //! Returns true, if setup() has been called
      inline const bool& is_setup() const { return issetup_; };

      //! Check if init() and setup() have been called
      virtual void check_init_setup() const;

      //! Check if init() has been called
      virtual void check_init() const;

     public:
      /*! \brief Returns the type of the current model evaluator
       *
       * @return Enum encoding the type of model evaluator
       */
      virtual Inpar::Solid::ModelType type() const = 0;

      /*! \brief Reset model specific variables (without jacobian)
       *
       *  This function is always called before the actual evaluate() routine is going to start.
       *  You can use it to reset model specific stuff at the beginning of a new evaluation round.
       *
       *  \param[in] x current full state vector
       *

       *  */
      virtual void reset(const Core::LinAlg::Vector<double>& x) = 0;

      /*! \brief Evaluate the current right-hand-side at \f$t_{n+1}\f$
       *

       *  */
      virtual bool evaluate_force() = 0;

      /*! \brief Evaluate the initial right hand side (overload if needed for specific model)
       *

       *  */
      virtual bool evaluate_initial_force() { return evaluate_force(); };

      /*! \brief Evaluate the current tangential stiffness matrix at \f$t_{n+1}\f$
       *

       *  */
      virtual bool evaluate_stiff() = 0;

      /*! \brief Evaluate the current right-hand-side vector and tangential stiffness matrix at
       * \f$t_{n+1}\f$
       *

       *  */
      virtual bool evaluate_force_stiff() = 0;

      /** \brief evaluate the right hand side for the cheap second order correction step
       *
       *  This is an optional method which is mainly considered for constraint
       *  models.
       *
       *  */
      virtual bool evaluate_cheap_soc_rhs() { return true; };

      /*! \brief Perform actions just before the evaluate() call
       *
       * Called in the very beginning of each call to one of the
       * Solid::ModelEvaluatorManager::Evaluate routines, such as evaluate_force,
       * evaluate_stiff, evaluate_force_stiff.
       *
       */
      virtual void pre_evaluate() = 0;

      /*! \brief Perform actions right after the evaluate() call
       *
       * Called at the end of each call to one of the
       * Solid::ModelEvaluatorManager::Evaluate routines, i.e. evaluate_force,
       * evaluate_stiff, evaluate_force_stiff.
       *
       */
      virtual void post_evaluate() = 0;

      /*! \brief Remove any condensed contributions from the structural rhs
       *
       * @param[in/out] rhs right-hand side vector
       *
       * */
      virtual void remove_condensed_contributions_from_rhs(Core::LinAlg::Vector<double>& rhs) {}

      /*! \brief Assemble the force right-hand-side
       *
       * After the evaluation of all models at the new state \f$t_{n+1}\f$ is finished, we
       * start to put everything together. At this point we build the right-hand-side at
       * the desired mid-point. The reason why we do it here and not at time integrator level
       * is that you have now the possibility to use a different time integration (not the
       * underlying structural one). This makes it more flexible. However, the contributions of the
       * old time step should be stored in the update_step_state() routine. There you can scale the
       * old contributions with the time integration factor you like and save them once, thus you
       * just have to add them to complete the mid-point right-hand-side.
       *
       * \param[out] f Right-hand side w/o viscous and/or mass effects scaled by time factor
       * \param[in] timefac_np Time factor of the underlying structural time integrator for
       *                       the new state at \f$t_{n+1}\f$.
       *
       * To scale the old state of the previous time step, see update_step_state.
       *
       * \return Boolean to indicate success (true) or error (false)
       *

       */
      virtual bool assemble_force(
          Core::LinAlg::Vector<double>& f, const double& timefac_np) const = 0;

      /*! \brief Assemble the jacobian
       *
       * \param[out] jac Jacobian matrix scaled by time integration factor
       * \param[in] timefac_np Time factor of the underlying structural time integrator for the new
       *                        state at \f$t_{n+1}\f$.
       *
       * \return Boolean to indicate success (true) or error (false)
       */
      virtual bool assemble_jacobian(
          Core::LinAlg::SparseOperator& jac, const double& timefac_np) const = 0;

      virtual bool assemble_cheap_soc_rhs(
          Core::LinAlg::Vector<double>& f, const double& timefac_np) const
      {
        return true;
      };

      /*! \brief write model specific restart
       *
       *  \param iowriter            (in) : output writer
       *  \param forced_writerestart (in) : special treatment is necessary, if the restart is forced
       */
      virtual void write_restart(
          Core::IO::DiscretizationWriter& iowriter, const bool& forced_writerestart) const = 0;

      /*! \brief read model specific restart information
       *
       *  \param ioreader (in) : input reader
       *

       *  */
      virtual void read_restart(Core::IO::DiscretizationReader& ioreader) = 0;

      /*! \brief Post setup operations
       */
      virtual void post_setup() {};

      /*! \brief predict the values for DoFs that are defined in
       *         the respective model evaluators, e.g. condensed variables.*/
      virtual void predict(const Inpar::Solid::PredEnum& pred_type) = 0;

      /*! \brief Recover condensed solution variables, meant to be called by run_post_compute_x
       */
      virtual void run_recover() {};

      /*! \brief Recover condensed solution variables
       *
       *  This method is supposed to be used to recover condensed solution variables.
       *  Typical examples are the EAS degrees of freedom or the dual Lagrange multipliers.
       *  Do NOT use it to reset your model variables! Use the reset() method instead.
       *
       *  */
      virtual void run_post_compute_x(const Core::LinAlg::Vector<double>& xold,
          const Core::LinAlg::Vector<double>& dir, const Core::LinAlg::Vector<double>& xnew) = 0;

      /*! \brief Executed before the solution vector is going to be updated
       *
       *  */
      virtual void run_pre_compute_x(const Core::LinAlg::Vector<double>& xold,
          Core::LinAlg::Vector<double>& dir_mutable, const NOX::Nln::Group& curr_grp) = 0;

      /*! \brief Executed at the end of the ::NOX::Solver::Generic::Step() (f.k.a. Iterate()) method
       *
       *  \param solver (in) : reference to the non-linear nox solver object (read-only)
       *
       *  */
      virtual void run_post_iterate(const ::NOX::Solver::Generic& solver) = 0;

      /*! \brief Executed at the beginning of the ::NOX::Solver::Generic::solve() method
       *
       *  \param solver (in) : reference to the non-linear nox solver object (read-only)
       *
       *  */
      virtual void run_pre_solve(const ::NOX::Solver::Generic& solver) {};

      /*! \brief Executed at the end of the NOX::Nln::LinearSystem::apply_jacobian_inverse()
       *  method
       *
       *  \note See comment in the NOX::Nln::PrePostOp::IMPLICIT::Generic class.
       *
       *  \param rhs   : read-only access to the rhs vector
       *  \param result: full access to the result vector of the linear system
       *  \param xold  : read-only access to the old x state vector
       *  \param grp   : read only access to the group object
       *
       *  */
      virtual void run_post_apply_jacobian_inverse(const Core::LinAlg::Vector<double>& rhs,
          Core::LinAlg::Vector<double>& result, const Core::LinAlg::Vector<double>& xold,
          const NOX::Nln::Group& grp)
      { /* empty */
      }


      /*! \brief Executed at the beginning of the NOX::Nln::LinearSystem::apply_jacobian_inverse()
       *  method
       *
       *  \note See comment in the NOX::Nln::PrePostOp::IMPLICIT::Generic class.
       *
       *  \param rhs   : read-only access to the rhs vector
       *  \param result: full access to the result vector of the linear system
       *  \param xold  : read-only access to the old x state vector
       *  \param grp   : read only access to the group object
       *
       *  */
      virtual void run_pre_apply_jacobian_inverse(const Core::LinAlg::Vector<double>& rhs,
          Core::LinAlg::Vector<double>& result, const Core::LinAlg::Vector<double>& xold,
          const NOX::Nln::Group& grp)
      { /* empty */
      }

      virtual bool correct_parameters(NOX::Nln::CorrectionType type) { return true; };

      /// update the model state corresponding to the time/load-step
      virtual void update_step_state(const double& timefac_n) = 0;

      // compute the element contributions for element based scaling using PTC
      virtual void evaluate_jacobian_contributions_from_element_level_for_ptc() {};
      // assemble the element contributions
      virtual void assemble_jacobian_contributions_from_element_level_for_ptc(
          std::shared_ptr<Core::LinAlg::SparseMatrix>& modjac, const double& timefac_n) {};

      //! Update the element by end of the time step
      virtual void update_step_element() = 0;

      //! Compute the residual by difference of {n+1} and {n} state
      virtual void update_residual() { /* do nothing by default */ }

      /*! \brief calculate the stress/strain contributions of each model evaluator
       *
       *  \remark This function is called from Solid::TimeInt::Base::prepare_output() and calculates
       *  missing quantities, which were not evaluated during the standard evaluate call and are
       *  only calculated once per load/time step. You can not do the calculations during the
       *  output_step_state() routine, because of the const status of the named function!
       *
       *  \sa output_step_state
       *
       *  */
      virtual void determine_stress_strain() = 0;

      /*! \brief calculate energy contributions of each model evaluator
       *
       *  \remark This function is called from Solid::TimeInt::Base::prepare_output() and calculates
       *  missing quantities, which were not evaluated during the standard evaluate call and are
       *  only calculated once per load/time step. You can not do the calculations during the
       *  output_step_state() routine, because of the const status of the named function!
       *
       *  \sa output_step_state
       *
       *  */
      virtual void determine_energy() = 0;

      /*! \brief calculate optional quantity contribution of each model evaluator
       *
       *  \remark This function is called from Solid::TimeInt::Base::prepare_output() and calculates
       *  missing quantities, which were not evaluated during the standard evaluate call and are
       *  only calculated once per load/time step. You can not do the calculations during the
       *  output_step_state() routine, because of the const status of the named function!
       *
       *  \sa output_step_state
       *
       *  */
      virtual void determine_optional_quantity() = 0;

      /*! \brief Output routine for model evaluator
       *
       * @param iowriter discretization writer that actually writes binary output to the disk
       */
      virtual void output_step_state(Core::IO::DiscretizationWriter& iowriter) const = 0;

      /**
       * \brief This method is called before the runtime output method is called.
       */
      virtual void runtime_pre_output_step_state() {};

      //! runtime output routine for model evaluator
      virtual void runtime_output_step_state() const {};

      //! reset routine for model evlaluator
      virtual void reset_step_state() = 0;

      //! post output routine for model evlaluator
      virtual void post_output() = 0;

      //! things that should be done after the timeloop
      virtual void post_time_loop() {};

      /** \brief Create a backup state
       *
       *  This is foremost meant for internally condensed quantities like for
       *  example the EAS state or the condensed Lagrange multiplier state. The
       *  global state in terms of the x-vector is stored more globally.
       */
      virtual void create_backup_state(const Core::LinAlg::Vector<double>& dir)
      {
        /* do nothing in default */
      }

      /** \brief Recover from the previously created backup state
       *
       *  This is foremost meant for internally condensed quantities like for
       *  example the EAS state or the condensed Lagrange multiplier state. The
       *  global state in terms of the x-vector is recovered more globally.
       *
       *  */
      virtual void recover_from_backup_state() { /* do nothing in default */ };

      //! @name Accessors to model specific things
      //! @{
      //! Returns a pointer to the model specific dof row map
      virtual std::shared_ptr<const Core::LinAlg::Map> get_block_dof_row_map_ptr() const = 0;

      /*! Returns a pointer to the current model solution vector (usually the
       *  Lagrange multiplier vector) */
      virtual std::shared_ptr<const Core::LinAlg::Vector<double>> get_current_solution_ptr()
          const = 0;

      /*! Returns a pointer to the model solution vector of the last time step
       *  (usually the Lagrange multiplier vector) */
      virtual std::shared_ptr<const Core::LinAlg::Vector<double>> get_last_time_step_solution_ptr()
          const = 0;

      /// access the current external load increment
      std::shared_ptr<Core::LinAlg::Vector<double>> get_fext_incr() const;

      //! Get the mechanical stress state vector (read access)
      [[nodiscard]] virtual std::shared_ptr<const Core::LinAlg::Vector<double>>
      get_mechanical_stress_state_np() const
      {
        return nullptr;
      }

      //! Get the mechanical stress state vector (read access)
      [[nodiscard]] virtual std::shared_ptr<const Core::LinAlg::Vector<double>>
      get_mechanical_stress_state_n() const
      {
        return nullptr;
      }

      //! @}

      //! @name internal accessors
      //! @{
      //! Returns the model evaluator data container
      const Solid::ModelEvaluator::Data& eval_data() const;

      //! Returns the global state data container
      const Solid::TimeInt::BaseDataGlobalState& global_state() const;

      //! Returns the global input/output data container
      const Solid::TimeInt::BaseDataIO& global_in_output() const;

      //! Returns the (structural) discretization
      const Core::FE::Discretization& discret() const;

      //! Returns the underlying Solid::Integrator object
      const Solid::Integrator& integrator() const;

      //! Returns the underlying Solid::TIMINT object
      const Solid::TimeInt::Base& tim_int() const;
      //! @}

     protected:
      /*! \brief Check the evaluation procedures for possible errors
       *
       *  In the standard case, we check for exceptions like overflow, invalid results
       *  or divide by zero operations. Furthermore, we look for an (optional) parameter
       *  named ele_eval_error_flag_. This is universal and should be usable by all model
       *  evaluators.
       *
       *  \return Boolean flag indicating success (true) or error (false)
       *
       */
      virtual bool eval_error_check() const;

      //! @name internal accessors
      //! @{
      //! Returns the model evaluator data container
      Solid::ModelEvaluator::Data& eval_data();
      std::shared_ptr<Solid::ModelEvaluator::Data>& eval_data_ptr();

      //! Returns the global state data container
      Solid::TimeInt::BaseDataGlobalState& global_state();
      std::shared_ptr<Solid::TimeInt::BaseDataGlobalState>& global_state_ptr();

      //! Returns the global input/output data container
      Solid::TimeInt::BaseDataIO& global_in_output();
      std::shared_ptr<Solid::TimeInt::BaseDataIO> global_in_output_ptr();

      //! Returns the (structural) discretization
      Core::FE::Discretization& discret();
      std::shared_ptr<Core::FE::Discretization>& discret_ptr();

      //! Returns the underlying Solid::Integrator object
      Solid::Integrator& integrator();
      std::shared_ptr<Solid::Integrator>& integrator_ptr();

      const int& dof_offset() const;
      //! @}
     protected:
      //! init flag
      bool isinit_;

      //! setup flag
      bool issetup_;

     private:
      //! pointer to the model evaluator data container
      std::shared_ptr<Solid::ModelEvaluator::Data> eval_data_ptr_;

      //! pointer to the global state data container
      std::shared_ptr<Solid::TimeInt::BaseDataGlobalState> gstate_ptr_;

      //! pointer to input/ouput data container
      std::shared_ptr<Solid::TimeInt::BaseDataIO> gio_ptr_;

      //! pointer to the problem discretization
      std::shared_ptr<Core::FE::Discretization> discret_ptr_;

      //! pointer to the structural (time) integrator
      std::shared_ptr<Solid::Integrator> int_ptr_;

      //! pointer to the time integrator strategy object
      std::shared_ptr<const Solid::TimeInt::Base> timint_ptr_;

      /*! \brief initial dof offset
       *
       *  This variable becomes important when you plan to create a
       *  block for the system of equations. E.g. saddle-point system
       *  in contact case. */
      int dof_offset_;

    };  // class Generic

  }  // namespace ModelEvaluator
}  // namespace Solid

FOUR_C_NAMESPACE_CLOSE

#endif
