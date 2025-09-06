// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_STRUCTURE_NEW_TIMINT_BASE_HPP
#define FOUR_C_STRUCTURE_NEW_TIMINT_BASE_HPP

#include "4C_config.hpp"

#include "4C_adapter_str_structure_new.hpp"
#include "4C_io_every_iteration_writer.hpp"
#include "4C_structure_new_timint_basedataglobalstate.hpp"
#include "4C_structure_new_timint_basedataio.hpp"
#include "4C_structure_new_timint_basedatasdyn.hpp"

// forward declaration
namespace Core::LinAlg
{
  template <typename T>
  class Vector;
  class Map;
}  // namespace Core::LinAlg


#include "4C_utils_parameter_list.fwd.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{
  class BlockSparseMatrixBase;
}  // namespace Core::LinAlg
namespace Adapter
{
  class StructureTimeAda;
}
namespace Solid
{
  class ModelEvaluatorManager;
  class Dbc;
  class Integrator;
  namespace ModelEvaluator
  {
    class Generic;
  }  // namespace ModelEvaluator
  namespace TimeInt
  {
    /** \brief Abstract class for all time integration strategies
     *
     *  */
    class Base : public Adapter::StructureNew, Core::IO::EveryIterationWriterInterface
    {
      friend class Adapter::StructureTimeAda;

     public:
      /// constructor
      Base();

      /// initialize (all already existing) class variables
      virtual void init(const std::shared_ptr<Solid::TimeInt::BaseDataIO> dataio,
          const std::shared_ptr<Solid::TimeInt::BaseDataSDyn> datasdyn,
          const std::shared_ptr<Solid::TimeInt::BaseDataGlobalState> dataglobalstate);

      /// setup of the new class variables
      void setup() override;

      /// run the post_setup tasks of the structural time integrator
      /// (e.g. compute mass matrix, initial accelerations, ...)
      void post_setup() override;

      /// tests if there are more time steps to do
      [[nodiscard]] bool not_finished() const override;

      /// reset everything (needed for biofilm simulations)
      void reset() override;

      /** \brief reset step configuration after time step
       *
       *  This function is supposed to reset all variables which are directly related
       *  to the current new step n+1. To be more precise all variables ending with "Np"
       *  have to be reset. */
      void reset_step() override;

      /// wrapper for things that should be done before prepare_time_step is called
      void pre_predict() override {}

      /// wrapper for things that should be done before solving the nonlinear iterations
      void pre_solve() override {}

      /// wrapper for things that should be done after convergence of Newton scheme
      void post_output() override {}

      /// things that should be done after the actual time loop is finished
      void post_time_loop() override;

      /// @name General access methods
      ///@{
      /// Access discretization (structure only)
      std::shared_ptr<Core::FE::Discretization> discretization() override;

      /// Access to pointer to DoF row map of the discretization (structure only)
      const Core::LinAlg::Map* dof_row_map_view() override
      {
        check_init();
        return dataglobalstate_->dof_row_map_view();
      }

      /// DoF map of structural vector of unknowns
      std::shared_ptr<const Core::LinAlg::Map> dof_row_map() override
      {
        check_init();
        return dataglobalstate_->dof_row_map();
      }

      //! DoF map of vector of unknowns
      //! Alternative method capable of multiple DoF sets
      std::shared_ptr<const Core::LinAlg::Map> dof_row_map(unsigned nds) override
      {
        check_init();
        return dataglobalstate_->dof_row_map(nds);
      }

      /// Access linear structural solver
      std::shared_ptr<Core::LinAlg::Solver> linear_solver() override
      {
        check_init();
        return datasdyn_->get_lin_solvers()[Inpar::Solid::model_structure];
      }

      /// Return MapExtractor for Dirichlet boundary conditions
      std::shared_ptr<const Core::LinAlg::MapExtractor> get_dbc_map_extractor() override;
      [[nodiscard]] std::shared_ptr<const Core::LinAlg::MapExtractor> get_dbc_map_extractor() const;

      //! Return locsys manager
      std::shared_ptr<Core::Conditions::LocsysManager> locsys_manager() override;

      //! Return the desired model evaluator (read-only)
      [[nodiscard]] const Solid::ModelEvaluator::Generic& model_evaluator(
          Inpar::Solid::ModelType mtype) const override;

      //! Return the desired model evaluator (read and write)
      Solid::ModelEvaluator::Generic& model_evaluator(Inpar::Solid::ModelType mtype) override;

      ///@}

      /// Return domain map of the mass matrix (implicit and explicit)
      [[nodiscard]] const Core::LinAlg::Map& get_mass_domain_map() const override;

      /// @name Coupled problem routines
      /// @{
      /// wrapper for things that should be done before updating
      void pre_update() override {}

      /// Update routine for coupled problems with monolithic approach
      void update() override;

      /// Update routine for coupled problems with monolithic approach with time adaptivity
      void update(double endtime) override = 0;

      /// Update time and step counter
      virtual void update_step_time();

      /// wrapper for things that should be done after solving the update
      void post_update() override;
      /// @}

      /// @name Access global state from outside via adapter (needed for coupled problems)
      ///@{
      /// unknown displacements at \f$t_{n+1}\f$
      [[nodiscard]] std::shared_ptr<const Core::LinAlg::Vector<double>> disp_np() const override
      {
        check_init();
        return dataglobalstate_->get_dis_np();
      }

      /* \brief write access to displacements at \f$t^{n+1}\f$
       *
       * Calling this method makes only sense if state is supposed
       * to be manipulated. We must not forget to synchronize the
       * manipulated state with the NOX group.
       * Otherwise, the manipulations will be overwritten by NOX.
       * Therefore, we set the flag state_is_insync_with_noxgroup_
       * to false.
       * This will be checked:
       * See \ref CheckStateInSyncWithNOXGroup
       *
       * See also \ref Adapter::StructureNew::set_state
       */
      std::shared_ptr<Core::LinAlg::Vector<double>> write_access_disp_np() override
      {
        check_init();
        set_state_in_sync_with_nox_group(false);
        return dataglobalstate_->get_dis_np();
      }

      /// known displacements at \f$t_{n}\f$
      [[nodiscard]] std::shared_ptr<const Core::LinAlg::Vector<double>> disp_n() const override
      {
        check_init();
        return dataglobalstate_->get_dis_n();
      }

      /// write access to displacements at \f$t^{n}\f$
      std::shared_ptr<Core::LinAlg::Vector<double>> write_access_disp_n() override
      {
        check_init();
        return dataglobalstate_->get_dis_n();
      }

      /// unknown velocities at \f$t_{n+1}\f$
      [[nodiscard]] std::shared_ptr<const Core::LinAlg::Vector<double>> vel_np() const override
      {
        check_init();
        return dataglobalstate_->get_vel_np();
      }

      /// write access to velocities at \f$t^{n+1}\f$
      std::shared_ptr<Core::LinAlg::Vector<double>> write_access_vel_np() override
      {
        check_init();
        return dataglobalstate_->get_vel_np();
      }

      /// unknown velocities at \f$t_{n}\f$
      [[nodiscard]] std::shared_ptr<const Core::LinAlg::Vector<double>> vel_n() const override
      {
        check_init();
        return dataglobalstate_->get_vel_n();
      }

      /// write access to velocities at \f$t^{n}\f$
      std::shared_ptr<Core::LinAlg::Vector<double>> write_access_vel_n() override
      {
        check_init();
        return dataglobalstate_->get_vel_n();
      }

      /// known velocities at \f$t_{n-1}\f$
      [[nodiscard]] std::shared_ptr<const Core::LinAlg::Vector<double>> vel_nm() const override
      {
        check_init();
        return dataglobalstate_->get_vel_nm();
      }

      /// unknown accelerations at \f$t_{n+1}\f$
      [[nodiscard]] std::shared_ptr<const Core::LinAlg::Vector<double>> acc_np() const override
      {
        check_init();
        return dataglobalstate_->get_acc_np();
      }

      //! known accelerations at \f$t_{n}\f$
      [[nodiscard]] std::shared_ptr<const Core::LinAlg::Vector<double>> acc_n() const override
      {
        check_init();
        return dataglobalstate_->get_acc_n();
      }
      ///@}

      /// @name access and modify model evaluator stuff via adapter
      /// @{
      /// are there any algebraic constraints?
      bool have_constraint() override
      {
        check_init_setup();
        return datasdyn_->have_model_type(Inpar::Solid::model_lag_pen_constraint);
      }

      /// FixMe get constraint manager defined in the structure
      std::shared_ptr<Constraints::ConstrManager> get_constraint_manager() override
      {
        FOUR_C_THROW("Not yet implemented!");
        return nullptr;
      }

      /// FixMe get contact/meshtying manager
      std::shared_ptr<CONTACT::MeshtyingContactBridge> meshtying_contact_bridge() override
      {
        FOUR_C_THROW("Not yet implemented!");
        return nullptr;
      }

      /// do we have this model
      bool have_model(Inpar::Solid::ModelType model) override
      {
        return datasdyn_->have_model_type(model);
      }

      /// Add residual increment to Lagrange multipliers stored in Constraint manager (derived)
      /// FixMe Different behavior for the implicit and explicit case!!!
      void update_iter_incr_constr(std::shared_ptr<Core::LinAlg::Vector<double>> lagrincr) override
      {
        FOUR_C_THROW("Not yet implemented!");
      }

      /// Add residual increment to pressures stored in Cardiovascular0D manager (derived)
      /// FixMe Different behavior for the implicit and explicit case!!!
      void update_iter_incr_cardiovascular0_d(
          std::shared_ptr<Core::LinAlg::Vector<double>> presincr) override
      {
        FOUR_C_THROW("Not yet implemented!");
      }
      /// @}

      /// @name Time step helpers
      ///@{
      /// Return current time \f$t_{n}\f$ (derived)
      [[nodiscard]] double get_time_n() const override
      {
        check_init();
        return dataglobalstate_->get_time_n();
      }

      /// Sets the current time \f$t_{n}\f$ (derived)
      void set_time_n(const double time_n) override
      {
        check_init();
        dataglobalstate_->get_time_n() = time_n;
      }

      /// Return target time \f$t_{n+1}\f$ (derived)
      [[nodiscard]] double get_time_np() const override
      {
        check_init();
        return dataglobalstate_->get_time_np();
      }

      /// Sets the target time \f$t_{n+1}\f$ of this time step (derived)
      void set_time_np(const double time_np) override
      {
        check_init();
        dataglobalstate_->get_time_np() = time_np;
      }

      /// Get upper limit of time range of interest (derived)
      [[nodiscard]] double get_time_end() const override
      {
        check_init();
        return datasdyn_->get_time_max();
      }

      /// Get upper limit of time range of interest (derived)
      void set_time_end(double timemax) override
      {
        check_init();
        datasdyn_->set_time_max(timemax);
      }

      /// Get time step size \f$\Delta t_n\f$
      [[nodiscard]] double get_delta_time() const override
      {
        check_init();
        return dataglobalstate_->get_delta_time()[0];
      }

      /// Set time step size \f$\Delta t_n\f$
      void set_delta_time(const double dt) override
      {
        check_init();
        dataglobalstate_->get_delta_time()[0] = dt;
      }

      /// Return time integration factor
      [[nodiscard]] double tim_int_param() const override;

      /// Return current step number \f$n\f$
      [[nodiscard]] int get_step_n() const override
      {
        check_init();
        return dataglobalstate_->get_step_n();
      }

      /// Sets the current step \f$n\f$
      void set_step_n(int step_n) override
      {
        check_init();
        dataglobalstate_->get_step_n() = step_n;
      }

      /// Return current step number $n+1$
      [[nodiscard]] int get_step_np() const override
      {
        check_init();
        return dataglobalstate_->get_step_np();
      }

      /// Sets the current step number \f$n+1\f$
      void set_step_np(int step_np) override
      {
        check_init_setup();
        dataglobalstate_->get_step_np() = step_np;
      }

      //! Get number of time steps
      [[nodiscard]] int get_step_end() const override
      {
        check_init();
        return datasdyn_->get_step_max();
      }

      /// Sets number of time steps
      void set_step_end(int step_end) override
      {
        check_init_setup();
        datasdyn_->set_step_max(step_end);
      }

      //! Get divcont type
      [[nodiscard]] virtual enum Inpar::Solid::DivContAct get_divergence_action() const
      {
        check_init_setup();
        return datasdyn_->get_divergence_action();
      }

      //! Get number of times you want to halve your timestep in case nonlinear solver diverges
      [[nodiscard]] virtual int get_max_div_con_refine_level() const
      {
        check_init_setup();
        return datasdyn_->get_max_div_con_refine_level();
      }

      //! Get random factor for time step adaption
      [[nodiscard]] virtual double get_random_time_step_factor() const
      {
        check_init_setup();
        return datasdyn_->get_random_time_step_factor();
      }

      //! Set random factor for time step adaption
      virtual double set_random_time_step_factor(double rand_tsfac)
      {
        check_init_setup();
        datasdyn_->set_random_time_step_factor(rand_tsfac);
        return datasdyn_->get_random_time_step_factor();
      }

      //! Get current refinement level for time step adaption
      [[nodiscard]] virtual int get_div_con_refine_level() const
      {
        check_init_setup();
        return datasdyn_->get_div_con_refine_level();
      }

      //! Set refinement level for time step adaption
      virtual int set_div_con_refine_level(int divconrefinementlevel)
      {
        check_init_setup();
        datasdyn_->set_div_con_refine_level(divconrefinementlevel);
        return datasdyn_->get_div_con_refine_level();
      }

      //! Get step of current refinement level for time step adaption
      [[nodiscard]] virtual int get_div_con_num_fine_step() const
      {
        check_init_setup();
        return datasdyn_->get_div_con_num_fine_step();
      }

      //! Set step of current refinement level for time step adaption
      virtual int set_div_con_num_fine_step(int divconnumfinestep)
      {
        check_init_setup();
        datasdyn_->set_div_con_num_fine_step(divconnumfinestep);
        return datasdyn_->get_div_con_num_fine_step();
      }

      /// set evaluation action
      void set_action_type(const Core::Elements::ActionType& action) override;

      // group id in nested parallelity
      [[nodiscard]] int group_id() const;
      ///@}

      /// Time adaptivity (derived pure virtual functionality)
      /// @{
      /// Resize MStep Object due to time adaptivity in FSI (derived)
      void resize_m_step_tim_ada() override;

      /// @}

      /// Output writer related routines (file and screen output)
      /// @{
      /// Access output object
      std::shared_ptr<Core::IO::DiscretizationWriter> disc_writer() override
      {
        return data_io().get_output_ptr();
      }

      /// Calculate all output quantities depending on the constitutive model
      /// (and, hence, on a potential material history)
      void prepare_output(bool force_prepare_timestep) override;

      /// output results (implicit and explicit)
      virtual void output() { output(false); }
      void output(bool forced_writerestart) override;

      /// Write Gmsh output for structural field
      void write_gmsh_struct_output_step() override;

      /// create result test for encapsulated structure algorithm
      std::shared_ptr<Core::Utils::ResultTest> create_field_test() override;

      /** \brief Get data that is written during restart
       *
       *  This routine is only for simple structure problems!

       *  */
      void get_restart_data(std::shared_ptr<int> step, std::shared_ptr<double> time,
          std::shared_ptr<Core::LinAlg::Vector<double>> disnp,
          std::shared_ptr<Core::LinAlg::Vector<double>> velnp,
          std::shared_ptr<Core::LinAlg::Vector<double>> accnp,
          std::shared_ptr<std::vector<char>> elementdata,
          std::shared_ptr<std::vector<char>> nodedata) override;

      /** Read restart values
       *
       * \param stepn (in): restart step at \f${n}\f$
       */
      void read_restart(const int stepn) override;

      /// Set restart values (deprecated)
      void set_restart(int stepn,  //!< restart step at \f${n}\f$
          double timen,            //!< restart time at \f$t_{n}\f$
          std::shared_ptr<Core::LinAlg::Vector<double>>
              disn,  //!< restart displacements at \f$t_{n}\f$
          std::shared_ptr<Core::LinAlg::Vector<double>>
              veln,  //!< restart velocities at \f$t_{n}\f$
          std::shared_ptr<Core::LinAlg::Vector<double>>
              accn,                                        //!< restart accelerations at \f$t_{n}\f$
          std::shared_ptr<std::vector<char>> elementdata,  //!< restart element data
          std::shared_ptr<std::vector<char>> nodedata      //!< restart element data
          ) override;
      /// @}

      /// Biofilm related stuff
      /// @{
      /// FixMe set structure displacement vector due to biofilm growth
      void set_str_gr_disp(
          std::shared_ptr<Core::LinAlg::Vector<double>> struct_growth_disp) override
      {
        FOUR_C_THROW("Currently unsupported!");
      }
      /// @}

      /// @name Pure virtual adapter functions (have to be implemented in the derived classes)
      /// @{
      /// integrate the current step (implicit and explicit)
      virtual int integrate_step() = 0;
      /// right-hand-side of Newton's method (implicit only)
      std::shared_ptr<const Core::LinAlg::Vector<double>> rhs() override { return get_f(); };
      [[nodiscard]] virtual std::shared_ptr<const Core::LinAlg::Vector<double>> get_f() const = 0;
      /// @}

     public:
      /// @name External accessors for the class variables
      ///@{
      /// Get the indicator if we are currently restarting the simulation
      [[nodiscard]] inline const bool& is_restarting() const { return isrestarting_; }

      /// Get the indicator if we need to restart the initial state
      [[nodiscard]] inline bool is_restarting_initial_state() const
      {
        return datasdyn_->is_restarting_initial_state();
      }

      /// Get TimIntBase data for global state quantities (read access)
      [[nodiscard]] std::shared_ptr<const BaseDataGlobalState> get_data_global_state_ptr() const
      {
        check_init();
        return dataglobalstate_;
      }

      /// Get TimIntBase data for global state quantities (read & write access)
      std::shared_ptr<BaseDataGlobalState>& get_data_global_state_ptr()
      {
        check_init();
        return dataglobalstate_;
      }

      [[nodiscard]] const BaseDataGlobalState& get_data_global_state() const
      {
        check_init();
        return *dataglobalstate_;
      }

      /// Get TimIntBase data for io quantities (read access)
      [[nodiscard]] std::shared_ptr<const BaseDataIO> get_data_io_ptr() const
      {
        check_init();
        return dataio_;
      }

      [[nodiscard]] const BaseDataIO& get_data_io() const
      {
        check_init();
        return *dataio_;
      }

      /// Get TimIntBase data or struct dynamics quantities (read access)
      [[nodiscard]] std::shared_ptr<const BaseDataSDyn> get_data_sdyn_ptr() const
      {
        check_init();
        return datasdyn_;
      }

      [[nodiscard]] const BaseDataSDyn& get_data_sdyn() const
      {
        check_init();
        return *datasdyn_;
      }

      /// return a reference to the Dirichlet Boundary Condition handler (read access)
      [[nodiscard]] const Solid::Dbc& get_dbc() const
      {
        check_init_setup();
        return *dbc_ptr_;
      }

      /// return a reference to the Dirichlet Boundary Condition handler (write access)
      Solid::Dbc& get_dbc()
      {
        check_init_setup();
        return *dbc_ptr_;
      }

      /// return a pointer to the Dirichlet Boundary Condition handler (read access)
      [[nodiscard]] std::shared_ptr<const Solid::Dbc> get_dbc_ptr() const
      {
        check_init_setup();
        return dbc_ptr_;
      }

      /// return the integrator (read-only)
      [[nodiscard]] const Solid::Integrator& integrator() const
      {
        check_init_setup();
        return *int_ptr_;
      }

      /// Get the global state
      const BaseDataGlobalState& data_global_state() const
      {
        check_init();
        return *dataglobalstate_;
      }

      /// Get internal TimIntBase data for structural dynamics quantities (read and write access)
      BaseDataSDyn& data_sdyn()
      {
        check_init();
        return *datasdyn_;
      }

      /// return a pointer to the Dirichlet Boundary Condition handler (read and write access)
      const std::shared_ptr<Solid::Dbc>& dbc_ptr()
      {
        check_init_setup();
        return dbc_ptr_;
      }

      [[nodiscard]] bool has_final_state_been_written() const override;

      /// get the indicator state
      [[nodiscard]] inline const bool& is_init() const { return isinit_; }

      /// get the indicator state
      [[nodiscard]] inline const bool& is_setup() const { return issetup_; }

      //! @name Attribute access functions
      //@{

      //! Provide Name
      virtual enum Inpar::Solid::DynamicType method_name() const = 0;

      //! Provide title
      std::string method_title() const;

      //! Return true, if time integrator is implicit
      virtual bool is_implicit() const = 0;

      //! Return true, if time integrator is explicit
      virtual bool is_explicit() const = 0;

      //! Provide number of steps, e.g. a single-step method returns 1,
      //! a \f$m\f$-multistep method returns \f$m\f$
      virtual int method_steps() const = 0;

      //! Give order of accuracy
      int method_order_of_accuracy() const
      {
        return std::min(method_order_of_accuracy_dis(), method_order_of_accuracy_vel());
      }

      //! Give local order of accuracy of displacement part
      virtual int method_order_of_accuracy_dis() const = 0;

      //! Give local order of accuracy of velocity part
      virtual int method_order_of_accuracy_vel() const = 0;

      //! Return linear error coefficient of displacements
      virtual double method_lin_err_coeff_dis() const = 0;

      //! Return linear error coefficient of velocities
      virtual double method_lin_err_coeff_vel() const = 0;

      //@}

      ///@}
     protected:
      /// Check if init() and setup() have been called, yet.
      inline void check_init_setup() const
      {
        FOUR_C_ASSERT(is_init() and is_setup(), "Call init() and setup() first!");
      }

      /// Check if init() has been called
      inline void check_init() const { FOUR_C_ASSERT(is_init(), "Call init() first!"); }

      /// Get the global state
      BaseDataGlobalState& data_global_state()
      {
        check_init();
        return *dataglobalstate_;
      }

      /// Get the pointer to global state
      const std::shared_ptr<BaseDataGlobalState>& data_global_state_ptr() const
      {
        check_init();
        return dataglobalstate_;
      }

      /// Get internal TimIntBase data for io quantities (read and write access)
      BaseDataIO& data_io()
      {
        check_init();
        return *dataio_;
      }

      /// return a pointer to the input/output data container (read and write access)
      const std::shared_ptr<BaseDataIO>& data_io_ptr()
      {
        check_init();
        return dataio_;
      }

      /// return a pointer to the structural dynamic data container (read and write access)
      const std::shared_ptr<BaseDataSDyn>& data_s_dyn_ptr()
      {
        check_init();
        return datasdyn_;
      }

      /// return a reference to the Dirichlet Boundary Condition handler (read and write access)
      Solid::Dbc& dbc()
      {
        check_init_setup();
        return *dbc_ptr_;
      }

      /// return a reference to the integrator (read and write access)
      Solid::Integrator& integrator()
      {
        check_init_setup();
        return *int_ptr_;
      }

      /// return a pointer to the integrator (read and write access)
      const std::shared_ptr<Solid::Integrator>& integrator_ptr()
      {
        check_init_setup();
        return int_ptr_;
      }

      /** \brief Output to file
       *
       *  This routine always prints the last converged state, i.e.
       *  \f$D_{n}, V_{n}, A_{n}\f$.
       *

       *  */
      void output_step(bool forced_writerestart);

     private:
      /*! \brief Create a new input/output step in the output writer
       *
       * New step is created only once per time step. This is controlled by \c datawritten.
       * Do nothing if data has already been written in this time step.
       *
       * \param[in,out] Indicator whether data has already been written in this time step (true) or
       *                not (false)
       */
      void new_io_step(bool& datawritten);

      /// output of the current state
      void output_state();

      /** \brief output of the current state */
      void output_state(Core::IO::DiscretizationWriter& iowriter, bool write_owner) const;

      /** \brief output of the debug state */
      void output_debug_state(
          Core::IO::DiscretizationWriter& iowriter, bool write_owner) const override;

      /// output during runtime
      void runtime_output_state() const;

      /// output reaction forces
      void output_reaction_forces();

      /// output energy
      void output_energy() const;

      /// write restart information
      void output_restart(bool& datawritten);

      /// add restart information to output state
      void add_restart_to_output_state();

      /** \brief set the number of nonlinear iterations of the last time step
       *
       *  \pre update_step_time() must be called beforehand, otherwise the wrong
       *  step-id is considered.
       *
       *  */
      void set_number_of_nonlinear_iterations();

      /** \brief decide which contributions to the total system energy shall be
       *         computed and written during simulation
       *
       *  */
      void select_energy_types_to_be_written();

      /** \brief initialize file stream for energy values and write all the
       *         column headers for the previously selected energy contributions
       *         to be written separately
       *
       *  */
      void initialize_energy_file_stream_and_write_headers();

     protected:
      /// flag indicating if init() has been called
      bool isinit_;

      /// flag indicating if setup() has been called
      bool issetup_;

      /// flag indicating that the simulation is currently restarting
      bool isrestarting_;

      /// flag indicating that displacement state was manipulated
      /// but NOX group has not been informed.
      bool state_is_insync_with_noxgroup_;

     protected:
      inline void set_state_in_sync_with_nox_group(const bool insync)
      {
        state_is_insync_with_noxgroup_ = insync;
      }

      inline void throw_if_state_not_in_sync_with_nox_group() const
      {
        if (!state_is_insync_with_noxgroup_)
        {
          FOUR_C_THROW(
              " state has been requested but the manipulated state has\n"
              "not been communicated to NOX.\n"
              "Manipulations made in the state vector will have no effect.\n"
              "Call set_state(x) to synchronize the states stored in the global\n"
              "state object and in the NOX group!");
        }
      }

     private:
      /// pointer to the different data containers
      std::shared_ptr<BaseDataIO> dataio_;
      std::shared_ptr<BaseDataSDyn> datasdyn_;
      std::shared_ptr<BaseDataGlobalState> dataglobalstate_;

      /// pointer to the integrator (implicit or explicit)
      std::shared_ptr<Solid::Integrator> int_ptr_;

      /// pointer to the dirichlet boundary condition handler
      std::shared_ptr<Solid::Dbc> dbc_ptr_;
    };  // class Base
  }  // namespace TimeInt
}  // namespace Solid

FOUR_C_NAMESPACE_CLOSE

#endif
