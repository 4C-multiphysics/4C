// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_THERMO_TIMINT_IMPL_HPP
#define FOUR_C_THERMO_TIMINT_IMPL_HPP

#include "4C_config.hpp"

#include "4C_coupling_adapter_mortar.hpp"
#include "4C_thermo_aux.hpp"
#include "4C_thermo_timint.hpp"

#include <Teuchos_Time.hpp>

FOUR_C_NAMESPACE_OPEN

namespace Adapter
{
  class CouplingMortar;
}

namespace Thermo
{
  //!
  //! \brief Front-end for thermal dynamics
  //!        with \b implicit time integration
  //!
  //! <h3> About </h3>
  //! The implicit time integrator object is a derivation of the base time integrators with an eye
  //! towards implicit time integration. #TimIntImpl provides the environment needed to execute
  //! implicit integrators. This is chiefly the non-linear solution technique, e.g., Newton-Raphson
  //! iteration. These iterative solution techniques require a set of control parameters which are
  //! stored within this object. It is up to derived object to implement the time-space discretised
  //! residuum an d its tangent. This object provides some utility functions to obtain various force
  //! vectors necessary in the calculation of the force residual in the derived time integrators.
  //!
  //! \author bborn
  //! \date 06/08
  class TimIntImpl : public TimInt
  {
   public:
    //! @name Construction
    //@{

    //! Constructor
    TimIntImpl(const Teuchos::ParameterList& ioparams,          //!< ioflags
        const Teuchos::ParameterList& tdynparams,               //!< input parameters
        const Teuchos::ParameterList& xparams,                  //!< extra flags
        std::shared_ptr<Core::FE::Discretization> actdis,       //!< current discretization
        std::shared_ptr<Core::LinAlg::Solver> solver,           //!< the solver
        std::shared_ptr<Core::IO::DiscretizationWriter> output  //!< the output
    );

    //@}

    //! build linear system tangent matrix, rhs/force residual
    //! Monolithic TSI accesses the linearised thermo problem
    void evaluate() override;

    //! @name Time step preparation
    //@{

    //! prepare time step
    void prepare_time_step() override;

    //! Predict target solution and identify residual
    void predict_step();

    //! Identify residual
    //! This method does not predict the target solution but
    //! evaluates the residual and the stiffness matrix.
    //! In partitioned solution schemes, it is better to keep the current
    //! solution instead of evaluating the initial guess (as the predictor)
    //! does.
    void prepare_step() override;

    //! Predict constant temperature, temperature rate,
    //! i.e. the initial guess is equal to the last converged step
    //! except Dirichlet BCs
    void predict_const_temp_rate();

    //! Predict constant temperature, however the rate
    //! is consistent to the time integration
    //! if the constant temperature is taken as correct temperature
    //! solution.
    //! This method has to be implemented by the individual time
    //! integrator.
    virtual void predict_const_temp_consist_rate() = 0;

    //! Predict temperature which satisfy exactly the Dirichlet BCs
    //! and the linearised system at the previously converged state.
    //!
    //! This is an implicit predictor, i.e. it calls the solver once.
    void predict_tang_temp_consist_rate();

    //@}

    //! @name Forces and tangents
    //@{

    //! Do residual force due to global balance of energy
    //! and its tangent with respect to the current
    //! temperatures \f$T_{n+1}\f$
    //!
    //! This is <i>the</i> central method which is different for each
    //! derived implicit time integrator. The time integrator implementation
    //! is expected to set members #fres_ and #tang_.
    //! The residual #fres_ is expected to follow the <i>same</i> sign
    //! convention like its tangent #tang_, i.e. to use
    //! Newton--Raphson's method the residual will be scaled by -1.
    virtual void evaluate_rhs_tang_residual() = 0;

    //@}

    //! @name Solution
    //@{

    //! determine characteristic norms for relative
    //! error checks of residual temperatures
    //! \author lw  \date 12/07
    virtual double calc_ref_norm_temperature() = 0;

    //! determine characteristic norms for relative
    //! error checks of residual forces
    //! \author lw  \date 12/07
    virtual double calc_ref_norm_force() = 0;

    //! Is convergence reached of iterative solution technique?
    //! Keep your fingers crossed...
    //! \author lw  \date 12/07
    bool converged();

    //! Solve dynamic equilibrium
    //!
    //! This is a general wrapper around the specific techniques.
    Thermo::ConvergenceStatus solve() override;

    //! Do full Newton-Raphson iteration
    //!
    //! This routines expects a prepared negative reisdual force #fres_
    //! and associated effective tangent matrix #tang_
    virtual Thermo::ConvergenceStatus newton_full();

    //! Blank Dirichlet dofs form residual and reactions
    //! calculate norms for convergence checks
    void blank_dirichlet_and_calc_norms();

    // check for success of nonlinear solve
    Thermo::ConvergenceStatus newton_full_error_check();

    //! Do (so-called) modified Newton-Raphson iteration in which
    //! the initial tangent is kept and not adapted to the current
    //! state of the temperature solution
    void newton_modified() { FOUR_C_THROW("Not impl."); }

    //! Prepare system for solving with Newton's method
    //!
    //! - negative residual
    //! - blank residual on Dirichlet DOFs
    //! - apply Dirichlet boundary conditions on system
    void prepare_system_for_newton_solve();

    //@}

    //! @name Updates
    //@{

    //! Update iteration
    //!
    //! This handles the iterative update of the current
    //! temperature \f$T_{n+1}\f$ with the residual temperature
    //! The temperature rate follow on par.
    void update_iter(const int iter  //!< iteration counter
    );

    //! Update iteration incrementally
    //!
    //! This update is carried out by computing the new #raten_
    //! from scratch by using the newly updated #tempn_. The method
    //! respects the Dirichlet DOFs which are not touched.
    //! This method is necessary for certain predictors
    //! (like #predict_const_temp_consist_rate)
    virtual void update_iter_incrementally() = 0;

    //! Update iteration incrementally with prescribed residual
    //! temperatures
    void update_iter_incrementally(const std::shared_ptr<const Core::LinAlg::Vector<double>>
            tempi  //!< input residual temperatures
    );

    //! Update iteration iteratively
    //!
    //! This is the ordinary update of #tempn_ and #raten_ by
    //! incrementing these vector proportional to the residual
    //! temperatures #tempi_
    //! The Dirichlet BCs are automatically respected, because the
    //! residual temperatures #tempi_ are blanked at these DOFs.
    virtual void update_iter_iteratively() = 0;

    //! Update configuration after time step
    //!
    //! This means, the state set
    //! \f$T_{n} := T_{n+1}\f$ and \f$R_{n} := R_{n+1}\f$
    //! Thus the 'last' converged state is lost and a reset
    //! of the time step becomes impossible.
    //! We are ready and keen awaiting the next time step.
    void update_step_state() override = 0;

    //! Update Element
    void update_step_element() override = 0;

    //! update time step
    void update() override;

    //! update Newton step
    void update_newton(std::shared_ptr<const Core::LinAlg::Vector<double>> tempi) override;

    //@}


    //! @name Output
    //@{

    //! Print to screen predictor information about residual norm etc.
    //! \author lw (originally) \date 12/07
    void print_predictor();

    //! Print to screen information about residual forces and temperatures
    //! \author lw (originally) \date 12/07
    void print_newton_iter();

    //! Contains text to print_newton_iter
    //! \author lw (originally) \date 12/07
    void print_newton_iter_text(FILE* ofile  //!< output file handle
    );

    //! Contains header to print_newton_iter
    //! \author lw (originally) \date 12/07
    void print_newton_iter_header(FILE* ofile  //!< output file handle
    );

    //! print summary after step
    void print_step() override;

    //@}

    //! @name Attribute access functions
    //@{

    //! Return time integrator name
    enum Thermo::DynamicType method_name() const override = 0;

    //@}

    //! @name Access methods
    //@{

    //! Return external force \f$F_{ext,n}\f$
    std::shared_ptr<Core::LinAlg::Vector<double>> fext() override = 0;

    //! Return external force \f$F_{ext,n+1}\f$
    virtual std::shared_ptr<Core::LinAlg::Vector<double>> fext_new() = 0;

    //! Return reaction forces
    //!
    //! This is a vector of length holding zeros at
    //! free DOFs and reaction force component at DOFs on DBCs.
    //! Mark, this is not true for DBCs with local coordinate
    //! systems in which the non-global reaction force
    //! component is stored in global Cartesian components.
    //! The reaction force resultant is not affected by
    //! this operation.
    std::shared_ptr<Core::LinAlg::Vector<double>> freact() override { return freact_; }

    //! Read and set external forces from file
    void read_restart_force() override = 0;

    //! Write internal and external forces for restart
    void write_restart_force(std::shared_ptr<Core::IO::DiscretizationWriter> output) override = 0;

    //! Return residual temperatures \f$\Delta T_{n+1}^{<k>}\f$
    std::shared_ptr<const Core::LinAlg::Vector<double>> temp_res() const { return tempi_; }

    //! initial guess of Newton's method
    std::shared_ptr<const Core::LinAlg::Vector<double>> initial_guess() override { return tempi_; }

    //! Set residual temperatures \f$\Delta T_{n+1}^{<k>}\f$
    void set_temp_residual(const std::shared_ptr<const Core::LinAlg::Vector<double>>
            tempi  //!< input residual temperatures
    )
    {
      if (tempi != nullptr) tempi_->Update(1.0, *tempi, 0.0);
    }

    //! Return effective residual force \f$R_{n+1}\f$
    std::shared_ptr<const Core::LinAlg::Vector<double>> force_res() const { return fres_; }

    //! right-hand side alias the dynamic force residual
    std::shared_ptr<const Core::LinAlg::Vector<double>> rhs() override { return fres_; }

    //@}

   protected:
    //! copy constructor is NOT wanted
    TimIntImpl(const TimIntImpl& old);

    // called when unconverged AND dicvont_halve_step
    void halve_time_step();

    void check_for_time_step_increase();

    //! @name General purpose algorithm parameters
    //@{
    enum Thermo::PredEnum pred_;  //!< predictor
    //@}

    //! @name Iterative solution technique
    //@{
    enum Thermo::NonlinSolTech itertype_;  //!< kind of iteration technique
                                           //!< or non-linear solution technique
    enum Thermo::ConvNorm normtypetempi_;  //!< convergence check for residual temperatures
    enum Thermo::ConvNorm normtypefres_;   //!< convergence check for residual forces

    enum Thermo::BinaryOp combtempifres_;  //!< binary operator to combine temperatures and forces

    enum Thermo::VectorNorm iternorm_;    //!< vector norm to check with
    int itermax_;                         //!< maximally permitted iterations
    int itermin_;                         //!< minimally requested iterations
    enum Thermo::DivContAct divcontype_;  // what to do when nonlinear solution fails
    int divcontrefinelevel_;              //!< refinement level of adaptive time stepping
    int divcontfinesteps_;  //!< number of time steps already performed at current refinement level
    double toltempi_;       //!< tolerance residual temperatures
    double tolfres_;        //!< tolerance force residual
    int iter_;              //!< iteration step
    int resetiter_;  //<! number of iterations already performed in resets of the current step
    double normcharforce_;  //!< characteristic norm for residual force
    double normchartemp_;   //!< characteristic norm for residual temperatures
    double normfres_;       //!< norm of residual forces
    double normtempi_;      //!< norm of residual temperatures
    std::shared_ptr<Core::LinAlg::Vector<double>> tempi_;  //!< residual temperatures
                                                           //!< \f$\Delta{T}^{<k>}_{n+1}\f$
    Teuchos::Time timer_;                                  //!< timer for solution technique
    std::shared_ptr<Coupling::Adapter::CouplingMortar>
        adaptermeshtying_;  //!< mortar coupling adapter
    //@}

    //! @name Various global forces
    //@{
    std::shared_ptr<Core::LinAlg::Vector<double>> fres_;    //!< force residual used for solution
    std::shared_ptr<Core::LinAlg::Vector<double>> freact_;  //!< reaction force
    //@}
  };
}  // namespace Thermo

FOUR_C_NAMESPACE_CLOSE

#endif
