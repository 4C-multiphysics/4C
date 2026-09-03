// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_ADAPTER_STR_STRUCTURE_HPP
#define FOUR_C_ADAPTER_STR_STRUCTURE_HPP


#include "4C_config.hpp"

#include "4C_adapter_field.hpp"
#include "4C_fem_general_elements_paramsinterface.hpp"
#include "4C_structure_new_input.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_parameter_list.fwd.hpp"
#include "4C_utils_result_test.hpp"

#include <cstdint>

FOUR_C_NAMESPACE_OPEN

namespace Solid
{
  /// Status of the current solid time-step.
  enum class StepStatus : std::int8_t
  {
    no_errors,                //< no errors occurred so far
    nonlinear_solver_failed,  //< nonlinear solver did not converge
    linear_solver_failed,     //< linear solver did not converge
    evaluation_failed,        //< evaluation of the residual or jacobian failed
  };

  /// Time-loop action selected from the current step status.
  enum class StepAction : std::int8_t
  {
    accept_step,  ///< accept the current step and continue normally
    retry_step,   ///< repeat the current step
  };
}  // namespace Solid

// forward declarations
namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Core::IO
{
  class DiscretizationWriter;
}

namespace Core::LinAlg
{
  class Solver;
  class SparseMatrix;
  class BlockSparseMatrixBase;
  class MapExtractor;
  class MultiMapExtractor;
  class Map;
}  // namespace Core::LinAlg

namespace Core::Conditions
{
  class LocsysManager;
}


namespace CONTACT
{
  class MeshtyingContactBridge;
}

namespace Constraints
{
  class ConstrManager;
  class SpringDashpotManager;
}  // namespace Constraints

namespace Utils
{
  class Cardiovascular0DManager;
}  // namespace Utils

namespace TimeInt
{
  template <typename>
  class TimIntMStep;
}

namespace Solid::ModelEvaluator
{
  class Generic;
}  // namespace Solid::ModelEvaluator

namespace Global
{
  class Problem;
}  // namespace Global

namespace Adapter
{
  /// general structural field interface
  /*!

  The point is to keep FSI as far apart from our field solvers as
  possible. Each structure field solver we want to use should get its own
  subclass of this. The FSI algorithm should be able to extract all the
  information from the structure field it needs using this interface.

  All FSI algorithms use this adapter to communicate with the structural
  field. There are different ways to use this adapter.

  In all cases you need to tell the structural algorithm about your time
  step. Therefore prepare_time_step(), update() and output() must be called at
  the appropriate position in the FSI algorithm.

  <h3>Dirichlet-Neumann coupled FSI</h3>

  A good starting displacement can be guessed with predict_interface_dispnp().

  Dirichlet-Neumann coupled FSI will need to Solve() the nonlinear
  structural problem for each time step after the fluid forces have been
  applied (apply_interface_forces()). Solve() will be called many times for each
  time step until the interface equilibrium is reached. The structural
  algorithm has to preserve its state until update() is called.

  After each Solve() you get the interface forces by extract_interface_dispnp().

  A Dirichlet-Neumann FSI with steepest descent relaxation or matrix free
  Newton Krylov will want to solve the structural problem linearly without
  history and prescribed interface forces: RelaxationSolve().

  <h3>Monolithic FSI</h3>

  Monolithic FSI is based on evaluate() of elements. This results in a new
  RHS() and a new SysMat(). Together with the initial_guess() these form the
  building blocks for a block based Newton's method.

  \warning Further cleanup is still needed.

  \sa Fluid, Ale

  */
  class Structure : public Field
  {
   public:
    //! @name Construction
    //@{

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
    virtual void setup() = 0;

    //@}

    /*!
     * @brief Perform all necessary tasks after setting up the object.
     */
    virtual void post_setup() = 0;

    //! @name Vector access
    //@{

    /// initial guess of Newton's method
    virtual std::shared_ptr<const Core::LinAlg::Vector<double>> initial_guess() = 0;

    /// rhs of Newton's method
    std::shared_ptr<const Core::LinAlg::Vector<double>> rhs() override = 0;

    /// unknown displacements at \f$t_{n+1}\f$
    [[nodiscard]] virtual std::shared_ptr<const Core::LinAlg::Vector<double>> dispnp() const = 0;

    /// known displacements at \f$t_{n}\f$
    [[nodiscard]] virtual std::shared_ptr<const Core::LinAlg::Vector<double>> dispn() const = 0;

    /// unknown velocity at \f$t_{n+1}\f$
    [[nodiscard]] virtual std::shared_ptr<const Core::LinAlg::Vector<double>> velnp() const = 0;

    /// known velocity at \f$t_{n}\f$
    [[nodiscard]] virtual std::shared_ptr<const Core::LinAlg::Vector<double>> veln() const = 0;

    /// known velocity at \f$t_{n-1}\f$
    [[nodiscard]] virtual std::shared_ptr<const Core::LinAlg::Vector<double>> velnm() const = 0;

    /// unknown acceleration at \f$t_{n+1}\f$
    [[nodiscard]] virtual std::shared_ptr<const Core::LinAlg::Vector<double>> accnp() const = 0;

    /// known acceleration at \f$t_{n}\f$
    [[nodiscard]] virtual std::shared_ptr<const Core::LinAlg::Vector<double>> accn() const = 0;

    virtual void resize_m_step_tim_ada() = 0;

    //@}

    //! @name Misc

    /// dof map of vector of unknowns
    std::shared_ptr<const Core::LinAlg::Map> dof_row_map() override = 0;

    /// DOF map of vector of unknowns for multiple dofsets
    virtual std::shared_ptr<const Core::LinAlg::Map> dof_row_map(unsigned nds) = 0;

    /// DOF map view of vector of unknowns
    virtual const Core::LinAlg::Map* dof_row_map_view() = 0;

    /// domain map of system matrix (do we really need this?)
    [[nodiscard]] virtual const Core::LinAlg::Map& domain_map() const = 0;

    /// direct access to system matrix
    std::shared_ptr<Core::LinAlg::SparseMatrix> system_matrix() override = 0;

    /// direct access to system matrix
    std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> block_system_matrix() override = 0;

    /// switch structure field to block matrix
    virtual void use_block_matrix(std::shared_ptr<const Core::LinAlg::MultiMapExtractor> domainmaps,
        std::shared_ptr<const Core::LinAlg::MultiMapExtractor> rangemaps) = 0;

    /// return contact/meshtying bridge
    virtual std::shared_ptr<CONTACT::MeshtyingContactBridge> meshtying_contact_bridge() = 0;

    /// do we have this model
    virtual bool have_model(Solid::ModelType model)
    {
      FOUR_C_THROW("new time integration only");
      return false;
    }

    /// return model evaluator
    virtual Solid::ModelEvaluator::Generic& model_evaluator(Solid::ModelType mtype) = 0;

    // access to locsys manager
    virtual std::shared_ptr<Core::Conditions::LocsysManager> locsys_manager() = 0;

    /// direct access to discretization
    virtual std::shared_ptr<Core::FE::Discretization> discretization() = 0;

    /// are there any algebraic constraints?
    virtual bool have_constraint() = 0;

    /// are there any spring dashpot bcs?
    virtual bool have_spring_dashpot() = 0;

    /// get constraint manager defined in the structure
    virtual std::shared_ptr<Constraints::ConstrManager> get_constraint_manager() = 0;

    /// get SpringDashpot manager defined in the structure
    virtual std::shared_ptr<Constraints::SpringDashpotManager> get_spring_dashpot_manager() = 0;

    /// Get type of thickness scaling for thin shell structures
    virtual Solid::StcScale get_stc_algo() = 0;

    /// Access to scaling matrix for STC
    virtual std::shared_ptr<Core::LinAlg::SparseMatrix> get_stc_mat() = 0;

    /// Return MapExtractor for Dirichlet boundary conditions
    virtual std::shared_ptr<const Core::LinAlg::MapExtractor> get_dbc_map_extractor() = 0;

    /// expand dirichlet bc map
    virtual void add_dirich_dofs(const std::shared_ptr<const Core::LinAlg::Map> maptoadd) {
      /* This is only needed for the old structural time integration.
         For the new structural time integration this is already
         implemented in str_dbc.cpp and str_dbc.H ! rauch 02/17 */
    };

    /// contract dirichlet bc map
    virtual void remove_dirich_dofs(const std::shared_ptr<const Core::LinAlg::Map> maptoremove) {
      /* This is only needed for the old structural time integration.
         For the new structural time integration this is already
         implemented in str_dbc.cpp and str_dbc.H ! rauch 02/17 */
    };

    /// set evaluation action
    virtual void set_action_type(const Core::Elements::ActionType& action) = 0;

    //@}

    //! @name Time step helpers
    //@{

    /// return time integration factor
    [[nodiscard]] virtual double tim_int_param() const = 0;

    //! Return current time \f$t_{n}\f$
    [[nodiscard]] virtual double time_old() const = 0;

    //! Return target time \f$t_{n+1}\f$
    [[nodiscard]] virtual double time() const = 0;

    /// Get upper limit of time range of interest
    [[nodiscard]] virtual double get_time_end() const = 0;

    //! Set upper limit of time range of interest
    virtual void set_time_end(double timemax) = 0;

    /// Get time step size \f$\Delta t_n\f$
    [[nodiscard]] virtual double dt() const = 0;

    /// Return current step number $n$
    [[nodiscard]] virtual int step_old() const = 0;

    /// Return current step number $n+1$
    [[nodiscard]] virtual int step() const = 0;

    /// Get number of time steps
    [[nodiscard]] virtual int num_step() const = 0;

    /// Take the time and integrate (time loop)

    virtual void integrate() = 0;

    /**
    @brief Converts the step status into a step action.

    A successful step status is directly converted into accept_step.
    If this function returns retry_step, it also is responsible for resetting the current step and
    preparing the next step for retry.

    @param step_status The status of the last solid evaluation or solve call.
    @return The action to take in the time loop.
     */
    virtual Solid::StepAction perform_error_action(Solid::StepStatus step_status)
    {
      // This base implementation only accepts successful steps and rejects all other statuses.
      // Derived classes may override this to implement retry mechanisms.
      if (step_status == Solid::StepStatus::no_errors) return Solid::StepAction::accept_step;
      FOUR_C_THROW("Structural solve failed.");
    }

    /// tests if there are more time steps to do
    [[nodiscard]] virtual bool not_finished() const = 0;

    /// start new time step
    void prepare_time_step() override = 0;

    /// set time step size
    virtual void set_dt(const double dtnew) = 0;

    //! Sets the current time \f$t_{n}\f$
    virtual void set_time(const double time) = 0;

    //! Sets the current step \f$n\f$
    virtual void set_step(int step) = 0;

    //! Sets the current step \f$n+1\f$
    virtual void set_stepn(int step) = 0;

    //! Sets the target time \f$t_{n+1}\f$ of this time step
    virtual void set_timen(const double time) = 0;

    /*!
    \brief update displacement and evaluate elements

    There are two displacement increments possible

    \f$x^n+1_i+1 = x^n+1_i + disiterinc\f$  (sometimes referred to as residual increment), and

    \f$x^n+1_i+1 = x^n     + disstepinc\f$

    with \f$n\f$ and \f$i\f$ being time and Newton iteration step

    Note: The structure expects an iteration increment.
    In case the StructureNOXCorrectionWrapper is applied, the step increment is expected
    which is then transformed into an iteration increment
    */
    void evaluate(std::shared_ptr<const Core::LinAlg::Vector<double>>
            disiterinc  ///< displacement increment between Newton iteration i and i+1
        ) override = 0;

    /// don't update displacement but evaluate elements (implicit only)
    virtual void evaluate() = 0;

    //! Calculate stresses and strains
    virtual void determine_stress_strain() = 0;

    /// update at time step end
    void update() override = 0;

    /// update at time step end in case of FSI time adaptivity
    virtual void update(double endtime) = 0;

    /// Update iteration
    /// Add residual increment to Lagrange multipliers stored in Constraint manager
    virtual void update_iter_incr_constr(
        std::shared_ptr<Core::LinAlg::Vector<double>> lagrincr) = 0;

    /// Access to output object
    virtual std::shared_ptr<Core::IO::DiscretizationWriter> disc_writer() = 0;

    /**
     * @brief Prepare output quantities (i.e., calculate stresses, strains, energies).
     * @note This has to be done before updating the state since otherwise a potential material
     * history is overwritten.
     *
     * @param force_prepare Flag to force output preparation outside the regular output schedule
     *          (e.g., for initial output).
     */
    void prepare_output(bool force_prepare_timestep) override = 0;

    // Get restart data
    virtual void get_restart_data(std::shared_ptr<int> step, std::shared_ptr<double> time,
        std::shared_ptr<Core::LinAlg::Vector<double>> disn,
        std::shared_ptr<Core::LinAlg::Vector<double>> veln,
        std::shared_ptr<Core::LinAlg::Vector<double>> accn,
        std::shared_ptr<std::vector<char>> elementdata,
        std::shared_ptr<std::vector<char>> nodedata) = 0;

    /// output results
    void output(bool forced_writerestart = false) override = 0;

    /// output results to screen
    virtual void print_step() = 0;

    /** \brief Finalize an accepted time step after update and output have been completed.
     *
     *  Derived methods may extend this hook with algorithm-specific work that belongs after
     *  accepting and writing a step, such as dynamic rebalancing or time-step control updates.
     */
    virtual void finalize_successful_step() { print_step(); }

    /// read restart information for given time step
    void read_restart(const int step) override = 0;

    /*!
    \brief Reset time step

    In case of time step size adaptivity, time steps might have to be repeated.
    Therefore, we need to reset the solution back to the initial solution of the
    time step.

    */
    virtual void reset_step() = 0;

    /// set restart information for parameter continuation
    virtual void set_restart(int step, double time,
        std::shared_ptr<Core::LinAlg::Vector<double>> disn,
        std::shared_ptr<Core::LinAlg::Vector<double>> veln,
        std::shared_ptr<Core::LinAlg::Vector<double>> accn,
        std::shared_ptr<std::vector<char>> elementdata,
        std::shared_ptr<std::vector<char>> nodedata) = 0;

    /// set the state of the nox group and the global state data container (implicit only)
    virtual void set_state(const std::shared_ptr<Core::LinAlg::Vector<double>>& x) = 0;

    /// wrapper for things that should be done before updating
    virtual void pre_update() = 0;

    /// wrapper for things that should be done after solving the update
    virtual void post_update() = 0;

    /// wrapper for things that should be done after the actual time loop is finished
    virtual void post_time_loop() = 0;

    //@}

    //! @name Solver calls

    /**
    @brief nonlinear solve

    Do the nonlinear solve, i.e. (multiple) corrector,
    for the time step. All boundary conditions have
    been set.

    @return the status of the nonlinear solve. Callers should check this status and take appropriate
    action if the solve was not successful.
     */
    [[nodiscard]] virtual Solid::StepStatus solve() = 0;

    /*!
    \brief linear structure solve with just a interface load

    The very special solve done in steepest descent relaxation
    calculation (and matrix free Newton Krylov).

    \note Can only be called after a valid structural solve.
    */
    virtual std::shared_ptr<Core::LinAlg::Vector<double>> solve_relaxation_linear() = 0;

    /// get the linear solver object used for this field
    virtual std::shared_ptr<Core::LinAlg::Solver> linear_solver() = 0;

    //@}

    //! @name Write access to field solution variables at \f$t^{n+1}\f$
    //@{

    /// write access to extract displacements at \f$t^{n+1}\f$
    virtual std::shared_ptr<Core::LinAlg::Vector<double>> write_access_dispnp() = 0;

    /// write access to extract velocities at \f$t^{n+1}\f$
    virtual std::shared_ptr<Core::LinAlg::Vector<double>> write_access_velnp() = 0;

    /// write access to extract displacements at \f$t^{n}\f$
    virtual std::shared_ptr<Core::LinAlg::Vector<double>> write_access_dispn() = 0;

    /// write access to extract velocities at \f$t^{n}\f$
    virtual std::shared_ptr<Core::LinAlg::Vector<double>> write_access_veln() = 0;

    //@}

    /// extract rhs (used to calculate reaction force for post-processing)
    virtual std::shared_ptr<Core::LinAlg::Vector<double>> freact() = 0;


    //! @name volume coupled specific methods
    //@{

    /// Set forces due to interface with fluid, the force is expected external-force-like
    ///
    /// \note This method will be deprecated as soon as new structural time integration is
    ///       completely engulfed by all algorithms using this method.
    virtual void set_force_interface(const Core::LinAlg::MultiVector<double>& iforce) = 0;

    //! specific method for iterative staggered partitioned TSI

    /// Identify residual
    /// This method does not predict the target solution but
    /// evaluates the residual and the stiffness matrix.
    /// In partitioned solution schemes, it is better to keep the current
    /// solution instead of evaluating the initial guess (as the predictor)
    /// does.
    //! will be obsolete after switch to new structural timint.
    virtual void prepare_partition_step() = 0;

    //@}

    /// create result test for encapsulated structure algorithm
    virtual std::shared_ptr<Core::Utils::ResultTest> create_field_test() = 0;

    /// reset time and state vectors (needed for biofilm growth simulations)
    virtual void reset() = 0;

    /// set structure displacement vector due to biofilm growth
    virtual void set_str_gr_disp(
        std::shared_ptr<Core::LinAlg::Vector<double>> struct_growth_disp) = 0;

    /// Write Gmsh output for structural field
    virtual void write_gmsh_struct_output_step() = 0;
  };


  /// structure field solver
  class StructureBaseAlgorithm
  {
   public:
    /// constructor
    StructureBaseAlgorithm(Global::Problem& problem, const Teuchos::ParameterList& prbdyn,
        const Teuchos::ParameterList& sdyn, std::shared_ptr<Core::FE::Discretization> actdis);

    /// virtual destructor to support polymorph destruction
    virtual ~StructureBaseAlgorithm() = default;

    /// structural field solver
    std::shared_ptr<Structure> structure_field() { return structure_; }

    /// structural field solver
    // std::shared_ptr<Solid::TimInt> StructureTimeIntegrator() { return str_tim_int_; }

   private:
    /// Create structure algorithm
    void create_structure(const Teuchos::ParameterList& prbdyn, const Teuchos::ParameterList& sdyn,
        std::shared_ptr<Core::FE::Discretization> actdis);

    /// setup structure algorithm of Solid::TimIntImpl type
    void create_tim_int(const Teuchos::ParameterList& prbdyn, const Teuchos::ParameterList& sdyn,
        std::shared_ptr<Core::FE::Discretization> actdis);

    /*! \brief Create linear solver for contact/meshtying problems
     *
     * Per default the CONTACT SOLVER block from the input file is used for generating the solver
     * object. The idea is, that this linear solver object is used whenever there is contact between
     * (two) structures. Otherwise the standard structural solver block is used (generated by
     * <tt>create_linear_solver</tt>. So we can use highly optimized solvers for symmetric pure
     * structural problems, but choose a different solver for the hard nonsymmetric contact case. We
     * automatically switch from the contact solver (in case of contact) to the structure solver
     * (pure structural problem, no contact) and back again.
     *
     * \note For contact/meshtying problems in the saddlepoint formulation (not condensed), this
     * routines requires a block preconditioner (eg <tt>CheapSIMPLE</tt>) as preconditioner for the
     * contact solver. The structure solver block specified in <tt>STRUCTURAL
     * DYNAMICS->LINEAR_SOLVER</tt>is used for the primary (structural) variables and the contact
     * solver block specified in <tt>CONTACT DYNAMIC->LINEAR_SOLVER</tt> is used for the
     * saddle-point system.
     *
     * \note Condensed meshtying problems use the standard structural solver block
     * (generated by create_linear_solver()). We assume that in contrary to contact problems,
     * the domain configuration is not changing for meshtying over the time.
     *
     * \param actdis discretization with all structural elements
     * \param[in] sdyn Structural parameters from input file
     *
     * \return Contact solver object
     *
     * \sa create_linear_solver()
     */
    std::shared_ptr<Core::LinAlg::Solver> create_contact_meshtying_solver(
        Core::FE::Discretization& actdis, const Teuchos::ParameterList& sdyn);

    /*! \brief Create linear solver for pure structure problems
     *
     * The solver block in the input file is specified by the parameter <tt>LINEAR_SOLVER</tt> in
     * the <tt>STRUCTURAL DYNAMICS</tt> block of the 4C input file. This solver is used for pure
     * structural problems, whenever there is no contact.
     *
     * To create the solver, we use the ID of the solver block to access the solver parameter list.
     * This is then used to create a Core::LinAlg::Solver.
     *
     * We also compute the nullspace information if this is required by the chosen solver.
     *
     * \param actdis discretization with all structural elements
     * \param[in] sdyn Structural parameters from input file
     *
     * \return Linear solver object for pure structural problems
     *
     * \sa create_contact_meshtying_solver()
     */
    std::shared_ptr<Core::LinAlg::Solver> create_linear_solver(
        std::shared_ptr<Core::FE::Discretization>& actdis, const Teuchos::ParameterList& sdyn);

    /// explicit global problem context
    Global::Problem& problem_;

    /// structural field solver
    std::shared_ptr<Structure> structure_;
  };

}  // namespace Adapter

FOUR_C_NAMESPACE_CLOSE

#endif
