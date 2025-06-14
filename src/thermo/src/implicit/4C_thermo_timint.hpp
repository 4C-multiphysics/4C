// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_THERMO_TIMINT_HPP
#define FOUR_C_THERMO_TIMINT_HPP

#include "4C_config.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_io_control.hpp"
#include "4C_io_discretization_visualization_writer_mesh.hpp"
#include "4C_io_runtime_csv_writer.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_thermo_adapter.hpp"
#include "4C_timestepping_mstep.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Thermo
{
  /*!
   * \brief Front-end for thermal dynamics by integrating in time.
   *
   * <h3> Intention </h3>
   * This front-end for thermal dynamics defines an interface to call
   * several derived time integrators. Thus it describes a plethora of pure
   * virtual methods which have to be implemented at the derived integrators.
   * However, it also offers a few non-empty methods and stores associated
   * data. The most important method of this base time integrator object
   * is #Integrate().
   *
   * #Integrate() performs a time integration (time loop) with constant
   * time steps and other parameters as set by the user.
   *
   * Although #Integrate is the main interface, this base time integrator
   * allows the public to access a few of its datum objects, for instance
   * the tangent system matrix #tang_ by #Tang(). This selective access
   * is needed in environments in which a independent time loop is provided.
   * This happens e.g. in fluid-structure-interaction.
   *
   * <h3> Responsibilities </h3>
   * Most importantly the base integrator manages the system state vectors and
   * matrices. It also deals with the output to files and offers method to
   * determine forces and tangents.
   */
  class TimInt : public Adapter
  {
   public:
    TimInt(const Teuchos::ParameterList& ioparams,              //!< ioflags
        const Teuchos::ParameterList& tdynparams,               //!< input parameters
        const Teuchos::ParameterList& xparams,                  //!< extra flags
        std::shared_ptr<Core::FE::Discretization> actdis,       //!< current discretisation
        std::shared_ptr<Core::LinAlg::Solver> solver,           //!< the solver
        std::shared_ptr<Core::IO::DiscretizationWriter> output  //!< the output
    );

    //! @name Actions
    //@{

    //! Equilibrate the initial state by identifying the consistent
    //! initial accelerations and (if applicable) internal variables
    //! Make capacity matrix
    void determine_capa_consist_temp_rate();

    //! Apply Dirichlet boundary conditions on provided state vectors
    void apply_dirichlet_bc(const double time,               //!< at time
        std::shared_ptr<Core::LinAlg::Vector<double>> temp,  //!< temperatures (may be nullptr)
        std::shared_ptr<Core::LinAlg::Vector<double>> rate,  //!< temperature rate (may be nullptr)
        bool recreatemap  //!< recreate mapextractor/toggle-vector
                          //!< which stores the DOF IDs subjected
                          //!< to Dirichlet BCs
                          //!< This needs to be true if the bounded DOFs
                          //!< have been changed.
    );

    //! prepare time step
    void prepare_time_step() override = 0;

    /// tests if there are more time steps to do
    bool not_finished() const override
    {
      return timen_ <= timemax_ + 1.0e-8 * (*dt_)[0] and stepn_ <= stepmax_;
    }

    //! non-linear solve
    //!
    //! Do the nonlinear solve, i.e. (multiple) corrector,
    //! for the time step. All boundary conditions have
    //! been set.
    Thermo::ConvergenceStatus solve() override = 0;

    //! build linear system tangent matrix, rhs/force residual
    //! Monolithic TSI accesses the linearised thermo problem
    void evaluate() override = 0;

    //! Update configuration after time step
    //!
    //! Thus the 'last' converged is lost and a reset of the time step
    //! becomes impossible. We are ready and keen awaiting the next
    //! time step.
    virtual void update_step_state() = 0;

    //! Update everything on element level after time step and after output
    //!
    //! Thus the 'last' converged is lost and a reset of the time step
    //! becomes impossible. We are ready and keen awaiting the next time step.
    virtual void update_step_element() = 0;

    //! Update time and step counter
    void update_step_time();

    //! update at time step end
    void update() override = 0;

    //! update Newton step
    void update_newton(std::shared_ptr<const Core::LinAlg::Vector<double>> tempi) override = 0;

    //! Reset configuration after time step
    //!
    //! Thus the last converged state is copied back on the predictor
    //! for current time step. This applies only to element-wise
    //! quantities
    void reset_step() override;

    //! set the initial thermal field
    void set_initial_field(const Thermo::InitialField,  //!< type of initial field
        const int startfuncno                           //!< number of spatial function
    );

    //@}

    //! @name Output
    //@{

    //! print summary after step
    void print_step() override = 0;

    //! runtime output write based on vtk, writing quantities like temperature, elemend ids, ...
    void write_runtime_output();

    //! Output to file
    //! This routine always prints the last converged state, i.e.
    //! \f$T_{n}, R_{n}\f$. So, #UpdateIncrement should be called
    //! upon object prior to writing stuff here.
    void output_step(bool forced_writerestart);

    //! output
    void output(bool forced_writerestart = false) override { output_step(forced_writerestart); }

    //! Write restart
    void output_restart(bool& datawritten  //!< (in/out) read and append if
                                           //!< it was written at this time step
    );

    //! Output temperatures, temperature rates
    //! and more system vectors
    void output_state(bool& datawritten  //!< (in/out) read and append if
                                         //!< it was written at this time step
    );

    //! Add restart information to output_state
    void add_restart_to_output_state();

    //! Heatflux & temperature gradient output
    void output_heatflux_tempgrad(bool& datawritten  //!< (in/out) read and append if
                                                     //!< it was written at this time step
    );

    //! Write internal and external forces (if necessary for restart)
    virtual void write_restart_force(std::shared_ptr<Core::IO::DiscretizationWriter> output) = 0;

    //! Identify residual
    //! This method does not predict the target solution but
    //! evaluates the residual and the stiffness matrix.
    //! In partitioned solution schemes, it is better to keep the current
    //! solution instead of evaluating the initial guess (as the predictor)
    //! does.
    void prepare_step() override = 0;

    //! thermal result test
    std::shared_ptr<Core::Utils::ResultTest> create_field_test() override;

    //@}

    //! @name Forces and Tangent
    //@{

    //! Apply external force
    void apply_force_external(const double time,                   //!< evaluation time
        const std::shared_ptr<Core::LinAlg::Vector<double>> temp,  //!< temperature state
        Core::LinAlg::Vector<double>& fext                         //!< external force
    );

    //! Apply convective boundary conditions force
    void apply_force_external_conv(Teuchos::ParameterList& p,
        const double time,                                          //!< evaluation time
        const std::shared_ptr<Core::LinAlg::Vector<double>> tempn,  //!< old temperature state T_n
        const std::shared_ptr<Core::LinAlg::Vector<double>> temp,   //!< temperature state T_n+1
        std::shared_ptr<Core::LinAlg::Vector<double>>& fext,        //!< external force
        std::shared_ptr<Core::LinAlg::SparseOperator> tang          //!< tangent at time n+1
    );

    //! Evaluate ordinary internal force, its tangent at state
    void apply_force_tang_internal(
        Teuchos::ParameterList& p,  //!< parameter list handed down to elements
        const double time,          //!< evaluation time
        const double dt,            //!< step size
        const std::shared_ptr<Core::LinAlg::Vector<double>> temp,   //!< temperature state
        const std::shared_ptr<Core::LinAlg::Vector<double>> tempi,  //!< residual temperatures
        std::shared_ptr<Core::LinAlg::Vector<double>> fint,         //!< internal force
        std::shared_ptr<Core::LinAlg::SparseMatrix> tang            //!< tangent matrix
    );

    //! Evaluate ordinary internal force, its tangent and the stored force at state
    void apply_force_tang_internal(
        Teuchos::ParameterList& p,  //!< parameter list handed down to elements
        const double time,          //!< evaluation time
        const double dt,            //!< step size
        const std::shared_ptr<Core::LinAlg::Vector<double>> temp,   //!< temperature state
        const std::shared_ptr<Core::LinAlg::Vector<double>> tempi,  //!< residual temperatures
        std::shared_ptr<Core::LinAlg::Vector<double>> fcap,         //!< capacity force
        std::shared_ptr<Core::LinAlg::Vector<double>> fint,         //!< internal force
        std::shared_ptr<Core::LinAlg::SparseMatrix> tang            //!< tangent matrix
    );

    //! Evaluate ordinary internal force
    //!
    //! We need incremental temperatures, because the internal variables,
    //! chiefly EAS parameters with an algebraic constraint, are treated
    //! as well. They are not treated perfectly, ie they are not iteratively
    //! equilibriated according to their (non-linear) constraint and
    //! the pre-determined temperatures -- we talk explicit time integration
    //! here, but they are applied in linearised manner. The linearised
    //! manner means the static condensation is applied once with
    //! residual temperatures replaced by the full-step temperature
    //! increment \f$T_{n+1}-T_{n}\f$.
    void apply_force_internal(
        Teuchos::ParameterList& p,  //!< parameter list handed down to elements
        const double time,          //!< evaluation time
        const double dt,            //!< step size
        const std::shared_ptr<Core::LinAlg::Vector<double>> temp,   //!< temperature state
        const std::shared_ptr<Core::LinAlg::Vector<double>> tempi,  //!< incremental temperatures
        std::shared_ptr<Core::LinAlg::Vector<double>> fint          //!< internal force
    );

    //@}

    //! @name Attributes
    //@{

    //! Provide Name
    virtual enum Thermo::DynamicType method_name() const = 0;

    //@}

    //! @name Access methods
    //@{

    //! Access dicretisation
    std::shared_ptr<Core::FE::Discretization> discretization() override { return discret_; }

    //! non-overlapping DOF map for multiple dofsets
    std::shared_ptr<const Core::LinAlg::Map> dof_row_map(unsigned nds) override
    {
      const Core::LinAlg::Map* dofrowmap = discret_->dof_row_map(nds);
      return std::make_shared<Core::LinAlg::Map>(*dofrowmap);
    }

    //! non-overlapping DOF map
    std::shared_ptr<const Core::LinAlg::Map> dof_row_map() override
    {
      const Core::LinAlg::Map* dofrowmap = discret_->dof_row_map();
      return std::make_shared<Core::LinAlg::Map>(*dofrowmap);
    }

    //! Access output object
    std::shared_ptr<Core::IO::DiscretizationWriter> disc_writer() override { return output_; }

    //! Read restart values at given restart step
    void read_restart(const int step) override;

    //! Read and set restart state
    void read_restart_state();

    //! Read and set restart forces
    virtual void read_restart_force() = 0;

    //! initial guess of Newton's method
    std::shared_ptr<const Core::LinAlg::Vector<double>> initial_guess() override = 0;

    //! Return temperatures \f$T_{n}\f$
    std::shared_ptr<Core::LinAlg::Vector<double>> tempn() override { return (*temp_)(0); }

    //! Return temperatures \f$T_{n+1}\f$
    std::shared_ptr<Core::LinAlg::Vector<double>> tempnp() override { return tempn_; }

    //! Return temperature rates \f$R_{n}\f$
    std::shared_ptr<Core::LinAlg::Vector<double>> raten() { return (*rate_)(0); }

    //! Return external force \f$F_{ext,n}\f$
    virtual std::shared_ptr<Core::LinAlg::Vector<double>> fext() = 0;

    //! Return reaction forces
    virtual std::shared_ptr<Core::LinAlg::Vector<double>> freact() = 0;

    //! right-hand side alias the dynamic thermal residual
    std::shared_ptr<const Core::LinAlg::Vector<double>> rhs() override = 0;

    //! Return tangent, i.e. thermal residual differentiated by temperatures
    //! (system_matrix()/stiff_ in STR)
    std::shared_ptr<Core::LinAlg::SparseMatrix> system_matrix() override { return tang_; }

    //! Return current time \f$t_{n}\f$
    double time_old() const override { return (*time_)[0]; }

    //! Return target time \f$t_{n+1}\f$
    double time() const override { return timen_; }

    //! Get upper limit of time range of interest
    double get_time_end() const override { return timemax_; }

    //! Get time step size \f$\Delta t_n\f$
    double dt() const override { return (*dt_)[0]; }

    //! Set time step size \f$\Delta t_n\f$
    void set_dt(double timestepsize) override { (*dt_)[0] = timestepsize; }

    //! Sets the target time \f$t_{n+1}\f$ of this time step
    void set_timen(const double time) override { timen_ = time; }

    //! Return current step number $n$
    int step_old() const override { return step_; }

    //! Return current step number $n+1$
    int step() const override { return stepn_; }

    //! Get number of time steps
    int num_step() const override { return stepmax_; }

    //! Update number of time steps (in adaptivity)
    virtual void set_num_step(const int newNumStep) { stepmax_ = newNumStep; }

    //! Return MapExtractor for Dirichlet boundary conditions
    std::shared_ptr<const Core::LinAlg::MapExtractor> get_dbc_map_extractor() override
    {
      return dbcmaps_;
    }

    //@}

   protected:
    //! evaluate error compared to analytical solution
    std::shared_ptr<std::vector<double>> evaluate_error_compared_to_analytical_sol();

    //! @name General purpose algorithm members
    //@{

    std::shared_ptr<Core::FE::Discretization> discret_;    //!< discretisation
    std::shared_ptr<Core::LinAlg::Solver> solver_;         //!< linear algebraic solver
    bool solveradapttol_;                                  //!< adapt solver tolerance
    double solveradaptolbetter_;                           //!< tolerance to which is adapted ????
    std::shared_ptr<Core::LinAlg::MapExtractor> dbcmaps_;  //!< map extractor object
                                                           //!< containing non-overlapping
                                                           //!< map of global DOFs on Dirichlet
                                                           //!< boundary conditions
    //@}

    //! @name Printing and output
    //@{

    std::shared_ptr<Core::IO::DiscretizationWriter> output_;  //!< binary output

    struct OptionsThermoVTKRuntimeOutput
    {
      /// whether to write pressure output
      bool output_temperature_state = false;

      /// whether to write heatflux output
      bool output_heatflux_state = false;

      /// whether to write temperature gradient output
      bool output_tempgrad_state = false;

      /// whether to write the owner of elements
      bool output_element_owner = false;

      /// whether to write the element GIDs
      bool output_element_gid = false;

      /// whether to write the node GIDs
      bool output_node_gid = false;
    };

    std::optional<Core::IO::DiscretizationVisualizationWriterMesh> runtime_vtk_writer_;
    OptionsThermoVTKRuntimeOutput runtime_vtk_params_;

    struct OptionsThermoCSVRuntimeOutput
    {
      /// whether to write energy output
      bool output_energy_state = false;
    };

    std::optional<Core::IO::RuntimeCsvWriter> runtime_csv_writer_;
    OptionsThermoCSVRuntimeOutput runtime_csv_params_;


    int printscreen_;        //!< print infos to standard out every n steps
    int writerestartevery_;  //!< write restart every given step;
                             //!< if 0, restart is not written
    bool writeglob_;         //!< write state on/off
    int writeglobevery_;     //!< write state every given step
    Thermo::HeatFluxType writeheatflux_;
    Thermo::TempGradType writetempgrad_;
    Thermo::CalcError calcerror_;  //!< evaluate error compared to analytical solution
    int errorfunctno_;             //!< function number of analytical solution for error evaluation
    //@}

    //! @name General time integration control parameters
    //@{

    std::shared_ptr<TimeStepping::TimIntMStep<double>>
        time_;      //!< time \f$t_{n}\f$ of last converged step
    double timen_;  //!< target time \f$t_{n+1}\f$
    std::shared_ptr<TimeStepping::TimIntMStep<double>> dt_;  //!< time step size \f$\Delta t\f$
    double timemax_;                                         //!< final time \f$t_\text{fin}\f$
    int stepmax_;                                            //!< final step \f$N\f$
    int step_;                                               //!< time step index \f$n\f$
    int stepn_;                                              //!< time step index \f$n+1\f$
    bool firstoutputofrun_;  //!< flag whether this output step is the first one (restarted or not)
    bool lumpcapa_;          //!< flag for lumping the capacity matrix, default: false

    //@}

    //! @name Global vectors
    //@{

    //! a zero vector of full length
    std::shared_ptr<Core::LinAlg::Vector<double>> zeros_;

    //@}

    //! @name Global state vectors
    //@{

    //! global temperatures \f${T}_{n}, T_{n-1}, ...\f$
    std::shared_ptr<TimeStepping::TimIntMStep<Core::LinAlg::Vector<double>>> temp_;

    //! global temperature rates \f${R}_{n}, R_{n-1}, ...\f$
    std::shared_ptr<TimeStepping::TimIntMStep<Core::LinAlg::Vector<double>>> rate_;

    //! global temperatures \f${T}_{n+1}\f$ at \f$t_{n+1}\f$
    std::shared_ptr<Core::LinAlg::Vector<double>> tempn_;

    //! global temperature rates \f${R}_{n+1}\f$ at \f$t_{n+1}\f$
    std::shared_ptr<Core::LinAlg::Vector<double>> raten_;

    //@}

    //! @name System matrices
    //@{

    //! holds eventually effective tangent (STR: stiff_)
    std::shared_ptr<Core::LinAlg::SparseMatrix> tang_;

    //@}

    std::shared_ptr<Core::LinAlg::MultiVector<double>>
        conductivity_;  //!< global conducitivity vector
  };
}  // namespace Thermo

FOUR_C_NAMESPACE_CLOSE

#endif
