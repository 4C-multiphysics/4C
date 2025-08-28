// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_STRUCTURE_TIMINT_IMPL_HPP
#define FOUR_C_STRUCTURE_TIMINT_IMPL_HPP

/*----------------------------------------------------------------------*/
/* headers */
#include "4C_config.hpp"

#include "4C_fem_condition.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_structure_timint.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{
  class MultiMapExtractor;
  class KrylovProjector;
}  // namespace Core::LinAlg

/*----------------------------------------------------------------------*/
/* belongs to structural dynamics namespace */
namespace Solid
{
  namespace Aux
  {
    class MapExtractor;
  }

  /*====================================================================*/
  /*!
   * \brief Front-end for structural dynamics
   *        with \b implicit time integration
   *
   * <h3> About </h3>
   * The implicit time integrator object is a derivation of the base time
   * integrators with an eye towards implicit time integration. #TimIntImpl
   * provides the environment needed to execute implicit integrators. This is
   * chiefly the non-linear solution technique, e.g., Newton-Raphson iteration.
   * These iterative solution techniques require a set of control parameters
   * which are stored within this object. It is up to derived object to
   * implement the time-space discretised residuum and its tangent. This object
   * provides some utility functions to obtain various force vectors necessary
   * in the calculation of the force residual in the derived time integrators.
   *

   */
  class TimIntImpl : public TimInt
  {
   public:
    //! @name Construction
    //@{

    //! Constructor
    TimIntImpl(const Teuchos::ParameterList& timeparams,
        const Teuchos::ParameterList& ioparams,               //!< ioflags
        const Teuchos::ParameterList& sdynparams,             //!< input parameters
        const Teuchos::ParameterList& xparams,                //!< extra flags
        std::shared_ptr<Core::FE::Discretization> actdis,     //!< current discretisation
        std::shared_ptr<Core::LinAlg::Solver> solver,         //!< the solver
        std::shared_ptr<Core::LinAlg::Solver> contactsolver,  //!< the solver for contact meshtying
        std::shared_ptr<Core::IO::DiscretizationWriter> output  //!< the output
    );

    /*! \brief Initialize this object

    Hand in all objects/parameters/etc. from outside.
    Construct and manipulate internal objects.

    \note Try to only perform actions in init(), which are still valid
          after parallel redistribution of discretizations.
          If you have to perform an action depending on the parallel
          distribution, make sure you adapt the affected objects after
          parallel redistribution.
          Example: cloning a discretization from another discretization is
          OK in init(...). However, after redistribution of the source
          discretization do not forget to also redistribute the cloned
          discretization.
          All objects relying on the parallel distribution are supposed to
          the constructed in \ref setup().

    \warning none
    \return bool

    */
    void init(const Teuchos::ParameterList& timeparams, const Teuchos::ParameterList& sdynparams,
        const Teuchos::ParameterList& xparams, std::shared_ptr<Core::FE::Discretization> actdis,
        std::shared_ptr<Core::LinAlg::Solver> solver) override;

    /*! \brief Setup all class internal objects and members

     setup() is not supposed to have any input arguments !

     Must only be called after init().

     Construct all objects depending on the parallel distribution and
     relying on valid maps like, e.g. the state vectors, system matrices, etc.

     Call all setup() routines on previously initialized internal objects and members.

    \note Must only be called after parallel (re-)distribution of discretizations is finished !
          Otherwise, e.g. vectors may have wrong maps.

    \warning none
    \return void

    */
    void setup() override;


    //! Resize \p TimIntMStep<T> multi-step quantities
    void resize_m_step() override = 0;

    //! return time integration factor
    double tim_int_param() const override = 0;

    //@}

    //! Do time integration of single step
    int integrate_step() override;

    //! Create Edges of for discrete shell elements
    void initialize_edge_elements();

    //! @name Prediction
    //@{

    //! Predict target solution and identify residual
    void predict();

    //! Identify residual
    //! This method does not predict the target solution but
    //! evaluates the residual and the stiffness matrix.
    //! In partitioned solution schemes, it is better to keep the current
    //! solution instead of evaluating the initial guess (as the predictor)
    //! does.
    void prepare_partition_step() override;

    //! Check if line search is applied in combination with elements
    //! that perform a local condensation (e.g. EAS)
    void prepare_line_search();

    //! Predict constant displacements, velocities and accelerations,
    //! i.e. the initial guess is equal to the last converged step
    //! except Dirichlet BCs
    void predict_const_dis_vel_acc();

    //! Predict constant displacements, however the velocities
    //! and accelerations are consistent to the time integration
    //! if the constant displacements are taken as correct displacement
    //! solution.
    //! This method has to be implemented by the individual time
    //! integrator.
    virtual void predict_const_dis_consist_vel_acc() = 0;

    //! Predict displacements based on the assumption of constant
    //! velocities. Calculate consistent velocities and accelerations
    //! afterwards.
    //! This method has to be implemented by the individual time
    //! integrator.
    virtual void predict_const_vel_consist_acc() = 0;

    //! Predict displacements based on the assumption of constant
    //! accelerations. Calculate consistent velocities and accelerations
    //! afterwards.
    //! This method has to be implemented by the individual time
    //! integrator.
    virtual void predict_const_acc() = 0;

    //! Predict displacements which satisfy exactly the Dirichlet BCs
    //! and the linearised system at the previously converged state.
    //!
    //! This is an implicit predictor, i.e. it calls the solver once.
    void predict_tang_dis_consist_vel_acc();

    //!
    void setup_krylov_space_projection(const Core::Conditions::Condition* kspcond);
    //!
    void update_krylov_space_projection();

    //@}

    /*! @name Forces
     *
     *  Apply all sets of forces (external, internal, damping, inertia, ...)
     *  and corresponding stiffnesses based on the current solution state.
     *
     *  On the level of Solid::TimIntImpl, we deal with forces and their stiffness
     *  contributions since an implicit time integration in 4C often requires
     *  full linearization.
     *
     *  For application of forces only (without stiffness), see Solid::TimInt.
     *
     *  \sa Solid::TimInt
     */
    //@{

    /*! \brief Evaluate forces, stiffness #stiff_ and residual #fres_
     *
     *  Do residual force due to global balance of momentum
     *  and its tangent with respect to the current
     *  displacements \f$D_{n+1}\f$
     *
     *  This is <i>the</i> central method which is different for each
     *  derived implicit time integrator. The time integrator implementation
     *  is expected to set members #fres_ and #stiff_.
     *  The residual #fres_ is expected to follow the <i>same</i> sign
     *  convention like its tangent #stiff_, i.e. to use
     *  Newton--Raphson's method the residual will be scaled by -1.
     *
     *  A parameter list is used to pass on additional information
     *  from from the non-linear solution procedure. For instance,
     *  in structural contact simulations and semi-smooth newton plasticity,
     *  it is of importance whether this method is called from a predictor step
     *  or from a regular Newton step.
     */
    virtual void evaluate_force_stiff_residual(Teuchos::ParameterList& params) = 0;

    /*! \brief Evaluate forces and residual #fres_
     *
     *  Do residual force due to global balance of momentum.
     *
     *  This is <i>the</i> central method which is different for each
     *  derived implicit time integrator. The time integrator implementation
     *  is expected to set members #fres_.
     *
     *  For now, this only provides basic functionalities, i.e. pure solid
     *  dynamics without contact, plasticity, constraints, ...
     *
     *  \f[ f_{res} = f_{int} - f_{ext} \f]
     *
     */
    virtual void evaluate_force_residual() = 0;

    //! Apply external force, its stiffness at state
    void apply_force_stiff_external(const double time,            //!< evaluation time
        const std::shared_ptr<Core::LinAlg::Vector<double>> dis,  //!< old displacement state
        Core::LinAlg::Vector<double>& disn,                       //!< new displacement state
        Core::LinAlg::Vector<double>& vel,                        // velocity state
        Core::LinAlg::Vector<double>& fext,                       //!< external force
        std::shared_ptr<Core::LinAlg::SparseOperator>& fextlin  //!< linearization of external force
    );

    //! Evaluate ordinary internal force, its stiffness at state
    void apply_force_stiff_internal(const double time,             //!< evaluation time
        const double dt,                                           //!< step size
        const std::shared_ptr<Core::LinAlg::Vector<double>> dis,   //!< displacement state
        const std::shared_ptr<Core::LinAlg::Vector<double>> disi,  //!< residual displacements
        Core::LinAlg::Vector<double>& vel,                         // velocity state
        std::shared_ptr<Core::LinAlg::Vector<double>> fint,        //!< internal force
        std::shared_ptr<Core::LinAlg::SparseOperator> stiff,       //!< stiffness matrix
        Teuchos::ParameterList& params,  //!< parameters from nonlinear solver
        std::shared_ptr<Core::LinAlg::SparseOperator> damp = nullptr  //!< material damping matrix
    );

    //! Evaluate internal and inertia forces and their linearizations
    void apply_force_stiff_internal_and_inertial(const double time,  //!< evaluation time
        const double dt,                                             //!< step size
        const double timintfac_dis,  //!< time integration factor for d(Res) / d dis
        const double timintfac_vel,  //!< time integration factor for d(Res) / d vel
        const std::shared_ptr<Core::LinAlg::Vector<double>> dis,   //!< displacement state
        const std::shared_ptr<Core::LinAlg::Vector<double>> disi,  //!< residual displacements
        const std::shared_ptr<Core::LinAlg::Vector<double>> vel,   //! velocity state
        const std::shared_ptr<Core::LinAlg::Vector<double>> acc,   //! acceleration state
        std::shared_ptr<Core::LinAlg::Vector<double>> fint,        //!< internal force
        std::shared_ptr<Core::LinAlg::Vector<double>> finert,      //!< inertia force
        std::shared_ptr<Core::LinAlg::SparseOperator> stiff,       //!< stiffness matrix
        std::shared_ptr<Core::LinAlg::SparseOperator> mass,        //!< mass matrix
        Teuchos::ParameterList& params,  //!< parameters from nonlinear solver
        const double beta = 1000.0,      //!< time integration parameters for element-wise time
                                     //!< integration (necessary for time integration of rotations)
        const double gamma = 1000.0,  //!< time integration parameters for element-wise time
                                      //!< integration (necessary for time integration of rotations)
        const double alphaf =
            1000.0,  //!< time integration parameters for element-wise time integration (necessary
                     //!< for time integration of rotations)
        const double alpham = 1000.0  //!< time integration parameters for element-wise time
                                      //!< integration (necessary for time integration of rotations)
    );

    //! Evaluate forces due to constraints
    void apply_force_stiff_constraint(const double time,           //!< evaluation time
        const std::shared_ptr<Core::LinAlg::Vector<double>> dis,   //!< last evaluated displacements
        const std::shared_ptr<Core::LinAlg::Vector<double>> disn,  //!< evaluation displacements
        std::shared_ptr<Core::LinAlg::Vector<double>>& fint,       //!< forces are added onto
        std::shared_ptr<Core::LinAlg::SparseOperator>& stiff,      //!< stiffness is added onto
        Teuchos::ParameterList pcon  //!< parameter list containing scale factors for matrix entries
    );

    //! Evaluate forces due to Cardiovascular0D BCs
    void apply_force_stiff_cardiovascular0_d(const double time,    //!< evaluation time
        const std::shared_ptr<Core::LinAlg::Vector<double>> disn,  //!< evaluation displacements
        std::shared_ptr<Core::LinAlg::Vector<double>>& fint,       //!< forces are added onto
        std::shared_ptr<Core::LinAlg::SparseOperator>& stiff,      //!< stiffness is added onto
        Teuchos::ParameterList
            pwindk  //!< parameter list containing scale factors for matrix entries
    );


    //! Evaluate forces and stiffness due to contact / meshtying
    void apply_force_stiff_contact_meshtying(
        std::shared_ptr<Core::LinAlg::SparseOperator>& stiff,  //!< stiffness is modified
        std::shared_ptr<Core::LinAlg::Vector<double>>& fres,   //!< residual forces are modified
        std::shared_ptr<Core::LinAlg::Vector<double>>& dis,    //!< current displacement state
        bool predict                                           //!< flag indicating predictor step
    );

    //! Evaluate forces and stiffness due to spring dash-pot boundary condition
    void apply_force_stiff_spring_dashpot(
        std::shared_ptr<Core::LinAlg::SparseOperator> stiff,  //!< stiffness is modified
        std::shared_ptr<Core::LinAlg::Vector<double>> fint,   //!< internal forces are modified
        std::shared_ptr<Core::LinAlg::Vector<double>> dis,    //!< current displacement state
        Core::LinAlg::Vector<double>& vel,                    //!< current velocity state
        bool predict,                                         //!< flag indicating predictor step
        Teuchos::ParameterList
            psprdash  //!< parameter list containing scale factors for matrix entries
    );

    //@}

    //! @name Solution
    //@{

    //! determine characteristic norms for relative
    //! error checks of residual displacements
    virtual double calc_ref_norm_displacement();

    //! determine characteristic norms for relative
    //! error checks of residual forces
    virtual double calc_ref_norm_force() = 0;

    //! Is convergence reached of iterative solution technique?
    //! Keep your fingers crossed...
    bool converged();

    /*!
    \brief Solve nonlinear dynamic equilibrium

    Do the nonlinear solve, i.e. (multiple) corrector,
    for the time step. All boundary conditions have
    been set.

    \return Enum to indicate convergence status or failure
    */
    Inpar::Solid::ConvergenceStatus solve() final;

    //! Do full Newton-Raphson iteration
    //!
    //! This routines expects a prepared negative residual force #fres_
    //! and associated effective stiffness matrix #stiff_
    // void NewtonFull();
    int newton_full();

    //! check for success of element evaluation in that no negative Jacobian
    //! determinant occurred, otherwise return error code
    int element_error_check(bool evalerr);

    //! check for success of linear solve otherwise return error code
    int lin_solve_error_check(int linerror);

    //! check for success of nonlinear solve otherwise return error code
    int newton_full_error_check(int linerror, int eleerror);

    //! Do (so-called) modified Newton-Raphson iteration in which
    //! the initial tangent is kept and not adapted to the current
    //! state of the displacement solution
    void newton_modified() { FOUR_C_THROW("Not impl."); }

    //! Do Line search iteration
    //!
    //! This routines expects a prepared negative residual force #fres_
    //! and associated effective stiffness matrix #stiff_
    int newton_ls();

    //! Solver call (line search)
    int ls_solve_newton_step();

    //! Update structural RHS and stiffness matrix (line search)
    void ls_update_structural_rh_sand_stiff(bool& isexcept, double& merit_fct);

    //! Evaluate the specified merit function (line search)
    //! (for pure structural problems this routine is rather short.
    //! However, to incorporate contact problems more easily the
    //! evaluation of the merit function is performed here.)
    //! return 0 if successful
    int ls_eval_merit_fct(double& merit_fct);

    //! Check the inner linesearch loop for convergence (line search)
    bool ls_converged(double* mf_value, double step_red);

    //! Print information concerning the last line search step (line search)
    void ls_print_line_search_iter(double* mf_value, int iter_ls, double step_red);

    //! Contains text to ls_print_line_search_iter
    void ls_print_ls_iter_text(FILE* ofile  //!< output file handle
    );

    //! Contains header to ls_print_line_search_iter
    void ls_print_ls_iter_header(FILE* ofile  //!< output file handle
    );

    //! Do classical augmented lagrange for volume constraint
    //!
    //! Potential is linearized wrt displacements
    //! keeping Lagrange multiplier fixed.
    //! Until convergence Lagrange multiplier
    //! is increased by Uzawa_param*(Vol_err)
    //!
    int uzawa_non_linear_newton_full();

    //! do full Newton iteration respecting volume constraint
    //!
    //! Potential is linearized wrt displacements
    //! and Lagrange multipliers
    //! Linear problem is solved with Uzawa algorithm.
    //!
    int uzawa_linear_newton_full();

    //! check for success of nonlinear solve otherwise return error code
    int uzawa_linear_newton_full_error_check(int linerror, int eleerror);

    //! Do pseudo transient continuation non-linear iteration
    //!
    //! Pseudo transient continuation is a variant of full newton that has a
    //! larger convergence radius than newton and is therefore more stable
    //! and/or can do larger time steps
    //!
    int ptc();

    //! Do nonlinear iteration for contact / meshtying
    //!
    int cmt_nonlinear_solve();

    /*! \brief Call linear solver for contact / meshtying
     *
     * We hold two distinct solver objects, #solver_ and #contactsolver_. Which one will be applied,
     * depends on the actual problem and the current status.
     *
     * <h3>Contact problems in saddle-point formulation</h3>
     * If no contact is active and contact hasn't been active neither in the last iteration nor the
     * last time step, we treat this as a pure structural problem and, thus, just apply the
     * structural solver #solver_.
     *
     * However, in case of contact, we use the #contactsolver_ to enable contact-specific solution
     * strategies suitable for the saddle-point problem.
     *
     * <h3>Problems without Lagrange multipliers or with Lagrange multipliers in condensed
     * formulation</h3>
     * For meshtying, we just apply the #contactsolver_.
     *
     * \todo Why do we use the #contactsolver_ instead of the #solver_? #solver_ should be
     * sufficient here since we don't need any meshtying-specific solution techniques in a condensed
     * formulation.
     *
     * For contact problems, where no contact is active and contact hasn't been active neither in
     * the last iteration nor the last time step, we use the regular structure solver #solver_. We
     * use the actual #contactsolver_ if contact is active.
     *
     * <h3>Problems with meshtying and contact</h3>
     * In this case, the meshtying contribution is always treated with dual Lagrange multipliers and
     * meshtying contributions will be condensed. Hence, the solver choice only depends on the
     * contact status. For details, see the section about solvers for contact problems.
     */
    void cmt_linear_solve();

    int cmt_windk_constr_nonlinear_solve();
    int cmt_windk_constr_linear_solve(const double k_ptc);

    //@}

    //! @name Updates
    //@{

    //! Update iteration
    //!
    //! This handles the iterative update of the current
    //! displacements \f$D_{n+1}\f$ with the residual displacements
    //! The velocities and accelerations follow on par.
    void update_iter(const int iter  //!< iteration counter
    );

    //! Update iteration incrementally
    //!
    //! This update is carried out by computing the new #veln_ and #accn_ from scratch by using the
    //! newly updated #disn_ according to the time integration scheme.
    //! We have to use this update routine if we are not sure whether velocities and accelerations
    //! have been computed consistently to the displacements based on time integration scheme
    //! specific formulas. Hence, this method is necessary for certain predictors (like
    //! #predict_const_dis_vel_acc).
    virtual void update_iter_incrementally() = 0;

    //! Update iteration incrementally with prescribed residual
    //! displacements
    void update_iter_incrementally(const std::shared_ptr<const Core::LinAlg::Vector<double>>
            disi  //!< input residual displacements
    );

    //! Update iteration iteratively
    //!
    //! This is the ordinary update of #disn_, #veln_ and #accn_ by
    //! incrementing these vector proportional to the residual
    //! displacements #disi_
    //! The Dirichlet BCs are automatically respected, because the
    //! residual displacements #disi_ are blanked at these DOFs.
    //! We can use this update routine if we are sure that the velocities
    //! and accelerations have been computed consistently to the displacements
    //! based on time integration scheme specific formulas.
    virtual void update_iter_iteratively() = 0;

    //! Update configuration after time step
    //!
    //! This means, the state set
    //! \f$D_{n} := D_{n+1}\f$, \f$V_{n} := V_{n+1}\f$ and \f$A_{n} := A_{n+1}\f$
    //! Thus the 'last' converged state is lost and a reset
    //! of the time step becomes impossible.
    //! We are ready and keen awaiting the next time step.
    void update_step_state() override = 0;

    //! Update Element
    void update_step_element() override = 0;

    //! Update step for constraints
    void update_step_constraint();

    //! Update step for Cardiovascular0D
    void update_step_cardiovascular0_d();

    //! Update step for SpringDashpot
    void update_step_spring_dashpot();

    //@}

    // export contact integration time and active set into text files
    // xxx.time and xxx.active
    void export_contact_quantities();

    //! @name Attribute access functions
    //@{

    //! Return time integrator name
    Inpar::Solid::DynamicType method_name() const override = 0;

    //! These time integrators are all implicit (mark their name)
    bool method_implicit() override { return true; }

    //! Provide number of steps, e.g. a single-step method returns 1,
    //! a m-multistep method returns m
    int method_steps() const override = 0;

    //! Give local order of accuracy of displacement part
    int method_order_of_accuracy_dis() const override = 0;

    //! Give local order of accuracy of velocity part
    int method_order_of_accuracy_vel() const override = 0;

    //! Return linear error coefficient of displacements
    double method_lin_err_coeff_dis() const override = 0;

    //! Return linear error coefficient of velocities
    double method_lin_err_coeff_vel() const override = 0;

    //! Return bool indicating if constraints are defined
    bool have_constraint() override;

    //! Return bool indicating if Cardiovascular0D bcs are defined
    bool have_cardiovascular0_d();

    //! Return bool indicating if spring dashpot BCs are defined
    bool have_spring_dashpot() override;

    //! Return Teuchos::rcp to ConstraintManager conman_
    std::shared_ptr<Constraints::ConstrManager> get_constraint_manager() override
    {
      return conman_;
    }

    //! Return Teuchos::rcp to Cardiovascular0DManager cardvasc0dman_
    std::shared_ptr<FourC::Utils::Cardiovascular0DManager> get_cardiovascular0_d_manager()
    {
      return cardvasc0dman_;
    }

    //! Return Teuchos::rcp to SpringDashpotManager springman_
    std::shared_ptr<Constraints::SpringDashpotManager> get_spring_dashpot_manager() override
    {
      return springman_;
    }

    //! Get type of thickness scaling for thin shell structures
    Inpar::Solid::StcScale get_stc_algo() override { return Inpar::Solid::StcScale::stc_inactive; }

    //! Update iteration
    //! Add residual increment to Lagrange multipliers stored in Constraint manager
    void update_iter_incr_constr(
        std::shared_ptr<Core::LinAlg::Vector<double>> lagrincr  ///< Lagrange multiplier increment
        ) override;

    //! Update iteration
    //! Add residual increment to pressures stored in Cardiovascular0D manager
    void update_iter_incr_cardiovascular0_d(
        std::shared_ptr<Core::LinAlg::Vector<double>> cv0ddofincr  ///< pressure increment
        ) override;

    //@}

    //! @name Fluid-structure-interaction specific methods
    //@{

    //! switch structure field to block matrix in fsi simulations
    void use_block_matrix(std::shared_ptr<const Core::LinAlg::MultiMapExtractor> domainmaps,
        std::shared_ptr<const Core::LinAlg::MultiMapExtractor> rangemaps) override;

    //! Evaluate/define the residual force vector #fres_ for
    //! relaxation solution with solve_relaxation_linear
    virtual void evaluate_force_stiff_residual_relax(Teuchos::ParameterList& params) = 0;

    //! Linear structure solve with just an interface load
    std::shared_ptr<Core::LinAlg::Vector<double>> solve_relaxation_linear() override;

    //! check, if according to divercont flag time step size can be increased
    void check_for_time_step_increase(Inpar::Solid::ConvergenceStatus& status);

    //! check, if according to divercont flag 3D0D PTC can be reset to normal Newton
    void check_for_3d0_dptc_reset(Inpar::Solid::ConvergenceStatus& status);

    /*! \brief Prepare system for solving with Newton's method
     *
     *  Blank DOFs with Dirichlet BCs in the residual. By default
     *  (preparejacobian = true), apply Dirichlet BCs to #stiff_ as well. This
     *  can be switched off when only the residual has been evaluated
     *  (\sa evaluate_force_residual()).
     */
    void prepare_system_for_newton_solve(
        const bool preparejacobian = true  ///< prepare Jacobian matrix
    );

    //@}

    //! @name Access methods
    //@{

    //! Return external force \f$F_{ext,n}\f$
    std::shared_ptr<Core::LinAlg::Vector<double>> fext() override = 0;

    //! Return external force \f$F_{ext,n+1}\f$
    std::shared_ptr<Core::LinAlg::Vector<double>> fext_new() override = 0;

    /*! \brief Return reaction forces
     *
     *  This is a vector of length holding zeros at free DOFs and reaction force
     *  component at DOFs on DBCs.
     *
     *  \note This is not true for DBCs with local coordinate systems in which
     *  the non-global reaction force  component is stored in global Cartesian
     *  components. The reaction force resultant is not affected by this
     *  operation.
     */
    std::shared_ptr<Core::LinAlg::Vector<double>> freact() override { return freact_; }

    //! Read and set external forces from file
    void read_restart_force() override = 0;

    //! Write internal and external forces for restart
    void write_restart_force(std::shared_ptr<Core::IO::DiscretizationWriter> output) override = 0;

    //! FORMERLY: Return residual displacements \f$\Delta D_{n+1}^{<k>}\f$
    //! Called from the previous adapters as initial_guess()
    std::shared_ptr<const Core::LinAlg::Vector<double>> initial_guess() override { return disi_; }

    //! Prepare time step
    void prepare_time_step() override;

    //! Update state incrementally for coupled problems with monolithic approach
    void update_state_incrementally(
        std::shared_ptr<const Core::LinAlg::Vector<double>> disiterinc) override
    {
      update_iter_incrementally(disiterinc);
      return;
    }

    //! Evaluate routine for coupled problems with monolithic approach
    void evaluate(std::shared_ptr<const Core::LinAlg::Vector<double>> disiterinc) override
    {
      update_iter_incrementally(disiterinc);

      Teuchos::ParameterList params;

      // builds tangent, residual and applies DBC
      evaluate_force_stiff_residual(params);
      prepare_system_for_newton_solve();
      return;
    }

    //! Update routine for coupled problems with monolithic approach
    void update() override
    {
      pre_update();
      update_step_state();
      update_step_time();
      update_step_element();
      post_update();
      return;
    }

    //! Update routine for coupled problems with monolithic approach with time adaptivity
    void update(const double endtime) override
    {
      pre_update();
      update_step_state();

      timen_ = endtime;

      // Update
      time_->update_steps(timen_);
      step_ = stepn_;
      stepn_ += 1;

      update_step_element();
      post_update();
      return;
    }

    //! Output results to binary file on disk
    void output(const bool forced_writerestart = false  ///< [in] Force writing of restart data
        ) override;

    //! Set residual displacements \f$\Delta D_{n+1}^{<k>}\f$
    void set_dis_residual(const std::shared_ptr<const Core::LinAlg::Vector<double>>
            disi  //!< input residual displacements
    )
    {
      if (disi != nullptr) disi_->update(1.0, *disi, 0.0);
    }

    //! Return the rhs-vector (negative sign for Newton is already included.)
    std::shared_ptr<const Core::LinAlg::Vector<double>> rhs() override { return fres_; }

    //@}

   protected:
    //! @name Output to file or screen
    //@{

    //! Print to screen predictor information about residual norm etc.
    void print_predictor();

    //! Print to screen information about residual forces and displacements
    void print_newton_iter();

    //! Contains text to print_newton_iter
    void print_newton_iter_text(FILE* ofile  //!< output file handle
    );

    //! Contains header to print_newton_iter
    void print_newton_iter_header(FILE* ofile  //!< output file handle
    );

    //! print statistics of converged Newton-Raphson iteration
    void print_newton_conv();

    //! print summary after step
    void print_step() override;

    //! The text for summary print, see #print_step
    void print_step_text(FILE* ofile  //!< output file handle
    );

    //@}

    //! copy constructor is NOT wanted
    TimIntImpl(const TimIntImpl& old);

    //! @name General purpose algorithm parameters
    //@{
    Inpar::Solid::PredEnum pred_;  //!< predictor
    //@}

    //! @name Iterative solution technique
    //@{
    enum Inpar::Solid::NonlinSolTech
        itertype_;  //!< kind of iteration technique or non-linear solution technique

    Inpar::Solid::ConvNorm normtypedisi_;        //!< convergence check for residual displacements
    Inpar::Solid::ConvNorm normtypefres_;        //!< convergence check for residual forces
    Inpar::Solid::ConvNorm normtypepres_;        //!< convergence check for residual pressure
    Inpar::Solid::ConvNorm normtypepfres_;       //!< convergence check for residual pressure forces
    Inpar::Solid::ConvNorm normtypecontconstr_;  //!< convergence check for contact constraints
                                                 //!< (saddlepoint formulation only)
    Inpar::Solid::ConvNorm normtypeplagrincr_;   //!< convergence check for Lagrange multiplier
                                                 //!< increment (saddlepoint formulation only)
    enum Inpar::Solid::BinaryOp
        combfresplconstr_;  //!< binary operator to combine field norms (forces and plastic
                            //!< constraints, semi-smooth plasticity only)
    enum Inpar::Solid::BinaryOp
        combdisiLp_;  //!< binary operator to combine field norms (displacement increments and Lp
                      //!< increments, semi-smooth plasticity only)
    enum Inpar::Solid::BinaryOp
        combfresEasres_;  //!< binary operator to combine field norms (forces
                          //!< and EAS residuals, semi-smooth plasticity only)
    enum Inpar::Solid::BinaryOp
        combdisiEasIncr_;  //!< binary operator to combine field norms (displacement increments and
                           //!< EAS increments, semi-smooth plasticity only)

    Inpar::Solid::BinaryOp combdispre_;     //!< binary operator to combine field norms
    Inpar::Solid::BinaryOp combfrespfres_;  //!< binary operator to combine field norms
    enum Inpar::Solid::BinaryOp
        combdisifres_;  //!< binary operator to combine displacement and forces
    enum Inpar::Solid::BinaryOp
        combfrescontconstr_;  //!< binary operator to combine field norms (forces and contact
                              //!< constraints, contact/meshtying in saddlepoint formulation only)
    enum Inpar::Solid::BinaryOp
        combdisilagr_;  //!< binary operator to combine field norms (displacement increments and LM
                        //!< increments, contact/meshtying in saddlepoint formulation only)

    Inpar::Solid::VectorNorm iternorm_;  //!< vector norm to check with
    int itermax_;                        //!< maximally permitted iterations
    int itermin_;                        //!< minimally requested iterations

    double toldisi_;        //!< tolerance residual displacements
    double tolfres_;        //!< tolerance force residual
    double tolpfres_;       //!< norm of residual pressure forces
    double tolpres_;        //!< norm of residual pressures
    double tolcontconstr_;  //!< norm of rhs for contact constraints (saddlepoint formulation only)
    double tollagr_;  //!< tolerance of LM multiplier increments (saddlepoint formulation only)

    double uzawaparam_;             //!< Parameter for Uzawa algorithm dealing
                                    //!< with Lagrange multipliers
    int uzawaitermax_;              //!< maximally permitted Uzawa iterations
    double tolcon_;                 //!< tolerance constraint
    double tolcardvasc0d_;          //!< tolerance for 0D cardiovascular residual
    double tolcardvasc0ddofincr_;   //!< tolerance for 0D cardiovascular dof increment
    int iter_;                      //!< iteration step
    double normcharforce_;          //!< characteristic norm for residual force
    double normchardis_;            //!< characteristic norm for residual displacements
    double normfres_;               //!< norm of residual forces
    double normfresr_;              //!< norm of reduced residual forces
    double normdisi_;               //!< norm of residual displacements
    double normdisir_;              //!< norm of reduced residual displacements
    double normcon_;                //!< norm of constraint
    double normcardvasc0d_;         //!< norm of 0D cardiovascular residual
    double normcardvasc0ddofincr_;  //!< norm of 0D cardiovascular dof increment
    double normpfres_;              //!< norm of residual pressure forces
    double normpres_;               //!< norm of residual pressures
    double normcontconstr_;  //!< norm of contact/meshtying constraint rhs (contact/meshtying in
                             //!< saddlepoint formulation only)
    double normlagr_;        //!< norm of lagrange multipliers
    double normw_;           //!< norm of wear
    double normwrhs_;
    double normwm_;  //!< norm of wear master
    double normwmrhs_;
    double alpha_ls_;    //!< line search step reduction
    double sigma_ls_;    //!< line search sufficient descent factor
    double ls_maxiter_;  //!< maximum number of line search steps
    double cond_res_;    //!< residual norm of condensed variables (e.g. EAS) needed for line search



    std::shared_ptr<Core::LinAlg::Vector<double>>
        disi_;  //!< residual displacements (and pressure) \f$\Delta{D}^{<k>}_{n+1}\f$

    //@}

    //! @name Various global forces
    //@{
    std::shared_ptr<Core::LinAlg::Vector<double>> fres_;    //!< force residual used for solution
    std::shared_ptr<Core::LinAlg::Vector<double>> freact_;  //!< reaction force
    //@}

    //! @name Krylov projection variables
    bool updateprojection_;  //!< bool triggering update of Krylov projection
    std::shared_ptr<Core::LinAlg::KrylovProjector> projector_;  //!< Krylov projector himself
    //@}

    //! @name Pseudo Transient Continuation Parameters
    //@{
    double ptcdt_;  //!< pseudo time step size for PTC
    double dti_;    //!< scaling factor for PTC (initially 1/ptcdt_, then adapted)
    //@}

  };  // class TimIntImpl

}  // namespace Solid

/*----------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
