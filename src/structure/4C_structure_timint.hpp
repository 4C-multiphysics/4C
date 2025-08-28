// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_STRUCTURE_TIMINT_HPP
#define FOUR_C_STRUCTURE_TIMINT_HPP

/*----------------------------------------------------------------------*/
/* headers */
#include "4C_config.hpp"

#include "4C_adapter_str_structure.hpp"
#include "4C_fem_general_elements_paramsinterface.hpp"
#include "4C_inpar_structure.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_timestepping_mstep.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>
#include <Teuchos_Time.hpp>

#include <fstream>
#include <iostream>
#include <memory>
#include <string>

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Core::FE
{
  class Discretization;
  class DiscretizationFaces;
}  // namespace Core::FE

namespace Utils
{
  class Cardiovascular0DManager;
}  // namespace Utils

namespace Constraints
{
  class ConstrManager;
  class ConstraintSolver;
  class SpringDashpotManager;
}  // namespace Constraints

namespace CONTACT
{
  class Beam3cmanager;
  class MeshtyingContactBridge;
}  // namespace CONTACT

namespace Core::LinAlg
{
  class Solver;
  class MapExtractor;
  class SparseMatrix;
  class SparseOperator;
  class BlockSparseMatrixBase;
}  // namespace Core::LinAlg

namespace Core::Conditions
{
  class LocsysManager;
}

namespace Core::IO
{
  class DiscretizationWriter;
}

namespace Cardiovascular0D
{
  class ProperOrthogonalDecomposition;
}

/*----------------------------------------------------------------------*/
namespace Solid
{
  /*====================================================================*/
  /*!
   * \brief Front-end for structural dynamics by integrating in time.
   *
   * <h3> Intention </h3>
   * This front-end for structural dynamics defines an interface to call
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
   * the tangent system matrix #stiff_ by #system_matrix(). This selective access
   * is needed in environments in which a independent time loop is provided.
   * This happens e.g. in fluid-structure-interaction.
   *
   * <h3> Responsibilities </h3>
   * Most importantly the base integrator manages the system state vectors and
   * matrices. It also deals with the output to files and offers method to
   * determine forces and stiffnesses (tangents).
   *

   */
  class TimInt : public Adapter::Structure
  {
    //! Structural time adaptivity is friend
    friend class TimAda;

    //! Joint auxiliary schemes are friends
    template <typename T>
    friend class TimAdaJoint;

   public:
    //! @name Life
    //@{

    //! Print tea time logo
    void logo();

    //! Constructor
    TimInt(const Teuchos::ParameterList& timeparams,
        const Teuchos::ParameterList& ioparams,               //!< ioflags
        const Teuchos::ParameterList& sdynparams,             //!< input parameters
        const Teuchos::ParameterList& xparams,                //!< extra flags
        std::shared_ptr<Core::FE::Discretization> actdis,     //!< current discretisation
        std::shared_ptr<Core::LinAlg::Solver> solver,         //!< the solver
        std::shared_ptr<Core::LinAlg::Solver> contactsolver,  //!< the solver for contact/meshtying
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
    virtual void init(const Teuchos::ParameterList& timeparams,
        const Teuchos::ParameterList& sdynparams, const Teuchos::ParameterList& xparams,
        std::shared_ptr<Core::FE::Discretization> actdis,
        std::shared_ptr<Core::LinAlg::Solver> solver);

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

    /*!
     * @brief Perform all necessary tasks after setting up the solid time
     * integration object.
     */
    void post_setup() override {}

    //! create fields, based on dofrowmap, whose previous time step values are unimportant
    virtual void create_fields();

    //! Construct all solution vectors
    void create_all_solution_vectors();

    //! Resize \p TimIntMStep<T> multi-step quantities
    virtual void resize_m_step() = 0;

    //! Resize \p TimIntMStep<T> multi-step quantities, needed for fsi time adaptivity
    void resize_m_step_tim_ada() override;

    //! Merge
    //!
    //! Merge basically duplicates the base object content of time
    //! integrator #tis onto the time integrator #this. This is like
    //! a copy, but a copy constructor is not permitted, because
    //! #TimInt is pure virtual.
    //! Usually this is not wanted when copying, but here it is
    //! highly appreciated. #TimInt contains only pointers (of the
    //! RefCount type) and can thus link -- or merge -- the data
    //! of #tis with #this. Practically, this turns up with time
    //! adaptivity in which #tis is the marching integrator
    //! and #this is the auxiliary method, which shares the marching data.
    void merge(TimInt& tis  //!< existing integrator to merge against
    )
    {
      // copy it
      *this = tis;

      return;
    }

    //@}

    //! @name Actions
    //@{

    //! Equilibrate the initial state by identifying the consistent
    //! initial accelerations and (if applicable) internal variables
    //! Make damping and mass matrix
    void determine_mass_damp_consist_accel();

    //! Clear mass matrix and evaluate mass matrix again.
    //! \note not implemented in base class.
    virtual void determine_mass();

    //! Apply Dirichlet boundary conditions on provided state vectors
    //! (reimplemented in static time integrator)
    virtual void apply_dirichlet_bc(const double time,      //!< at time
        std::shared_ptr<Core::LinAlg::Vector<double>> dis,  //!< displacements
                                                            //!< (may be nullptr)
        std::shared_ptr<Core::LinAlg::Vector<double>> vel,  //!< velocities
                                                            //!< (may be nullptr)
        std::shared_ptr<Core::LinAlg::Vector<double>> acc,  //!< accelerations
                                                            //!< (may be nullptr)
        bool recreatemap  //!< recreate map extractor/toggle-vector
                          //!< which stores the DOF IDs subjected
                          //!< to Dirichlet BCs
                          //!< This needs to be true if the bounded DOFs
                          //!< have been changed.
    );

    /// start new time step
    void prepare_time_step() override = 0;

    //! Do time integration of multiple steps
    int integrate() override
    {
      FOUR_C_THROW("time loop moved to separate adapter");
      return 0;
    }

    /// tests if there are more time steps to do
    bool not_finished() const override
    {
      return timen_ <= timemax_ + 1.0e-8 * (*dt_)[0] and stepn_ <= stepmax_;
    }

    //! do something in case nonlinear solution does not converge for some reason
    Inpar::Solid::ConvergenceStatus perform_error_action(
        Inpar::Solid::ConvergenceStatus nonlinsoldiv) override;

    //! Do time integration of single step
    virtual int integrate_step() = 0;


    /*! \brief Non-linear solve
     *
     *  Do the nonlinear solve, i.e. (multiple) corrector,
     *  for the time step. All boundary conditions have been set.
     */
    Inpar::Solid::ConvergenceStatus solve() override = 0;

    //! Linear structure solve with just an interface load
    std::shared_ptr<Core::LinAlg::Vector<double>> solve_relaxation_linear() override = 0;

    /*! \brief Update displacement in case of coupled problems
     *
     *  We always need iterative displacement increments here:
     *
     *  x^n+1_i+1 = x^n+1_i + disiterinc (sometimes referred to as residual increment)
     *
     *  with n and i being time and Newton iteration step
     */
    void update_state_incrementally(
        std::shared_ptr<const Core::LinAlg::Vector<double>> disiterinc) override = 0;

    /*! \brief Update displacement and evaluate elements in case of coupled problems
     *
     *  We always need iterative displacement increments here:
     *
     *  x^n+1_i+1 = x^n+1_i + disiterinc (sometimes referred to as residual increment)
     *
     *  with n and i being time and Newton iteration step
     */
    void evaluate(std::shared_ptr<const Core::LinAlg::Vector<double>> disiterinc) override = 0;

    /// don't update displacement but evaluate elements (implicit only)
    void evaluate() override { FOUR_C_THROW("new structural time integration only"); }

    /// update at time step end
    void update() override = 0;

    /// update at time step end in case of FSI time adaptivity
    void update(const double endtime) override = 0;

    /// Update iteration
    /// Add residual increment to Lagrange multipliers stored in Constraint manager
    void update_iter_incr_constr(
        std::shared_ptr<Core::LinAlg::Vector<double>> lagrincr  ///< Lagrange multiplier increment
        ) override = 0;

    /// output results
    void output(bool forced_writerestart = false) override = 0;

    //! Update configuration after time step
    //!
    //! Thus the 'last' converged is lost and a reset of the time step
    //! becomes impossible. We are ready and keen awaiting the next
    //! time step.
    virtual void update_step_state() = 0;

    //! Update everything on element level after time step and after output
    //!
    //! Thus the 'last' converged is lost and a reset of the time step
    //! becomes impossible. We are ready and keen awaiting the next
    //! time step.
    virtual void update_step_element() = 0;

    //! Update time and step counter
    void update_step_time();

    //! Update step for contact / meshtying
    void update_step_contact_meshtying();

    //! Velocity update method (VUM) for contact
    //!
    //! The VUM is an explicit update method at the end of each time step
    //! which is supposed to assure exact algorithmic conservation of total
    //! energy during contact. Further details can be found in the original
    //! paper by Laursen and Love (IJNME, 2002) and in the more recent and
    //! mortar-related paper by Hartmann et al. (IJNME, 2007).
    //! CAUTION: The VUM is only available for GenAlpha and GEMM.
    void update_step_contact_vum();

    //! Reset configuration after time step
    //!
    //! Thus the last converged state is copied back on the predictor
    //! for current time step. This applies only to element-wise
    //! quantities
    void reset_step() override;

    //! Set initial fields in structure (e.g. initial velocities)
    void set_initial_fields();

    //@}

    //! @name Determination of output quantities that depend on
    // the constitutive model
    //@{
    //! Calculate all output quantities depending on the constitutive model
    //  (and, hence, on a potential material history)
    void prepare_output(bool force_prepare_timestep) override;

    //! Calculate stresses, strains on micro-scale
    void prepare_output_micro();

    //! Calculate stresses and strains
    void determine_stress_strain() override;

    //! Calculate kinetic, internal and external energy
    virtual void determine_energy();

    /// create result test for encapsulated structure algorithm
    std::shared_ptr<Core::Utils::ResultTest> create_field_test() override;

    //@}


    //! @name Output
    //@{

    //! print summary after step
    void print_step() override = 0;

    //! Output to file
    //! This routine always prints the last converged state, i.e.
    //! \f$D_{n}, V_{n}, A_{n}\f$. So, #UpdateIncrement should be called
    //! upon object prior to writing stuff here.
    void output_step(const bool forced_writerestart = false  ///< [in] Force writing of restart data
    );


    //! Write output for every Newton or line search iteration
    //! The step numbers are formatted in the following manner:
    //!  n    5               4 2                     1 0
    //!  00..00               000                     00
    //! |__ ___|             |_ _|                   |_ |
    //!    V                   V                       V
    //! digits n to 5       digits 4 to 2            digits 1 to 0
    //! represent the       represent the            represent the
    //! time steps          Newton steps             line search steps
    void output_every_iter(bool nw = false, bool ls = false);

    //! write output of step to the Gmsh format
    void write_gmsh_struct_output_step() override;

    //! Write restart
    virtual void output_restart(bool& datawritten  //!< (in/out) read and append if
                                                   //!< it was written at this time step
    );
    //! Get data that is written during restart
    void get_restart_data(std::shared_ptr<int> step, std::shared_ptr<double> time,
        std::shared_ptr<Core::LinAlg::Vector<double>> disn,  //!< new displacement state
        std::shared_ptr<Core::LinAlg::Vector<double>> veln,  //!< new velocity state
        std::shared_ptr<Core::LinAlg::Vector<double>> accn,  //!< new acceleration state
        std::shared_ptr<std::vector<char>>
            elementdata,  //!< internal element/history variables e.g. F_prestress
        std::shared_ptr<std::vector<char>> nodedata  //
        ) override;

    //! Output displacements, velocities and accelerations
    //! and more system vectors
    virtual void output_state(bool& datawritten  //!< (in/out) read and append if
                                                 //!< it was written at this time step
    );

    //! Add restart information to output_state
    void add_restart_to_output_state();

    //! Stress & strain output
    void output_stress_strain(bool& datawritten  //!< (in/out) read and append if
                                                 //!< it was written at this time step
    );

    //! Energy output
    void output_energy();

    //! Active set, energy and momentum output for contact
    void output_contact();

    //! Write internal and external forces (if necessary for restart)
    virtual void write_restart_force(std::shared_ptr<Core::IO::DiscretizationWriter> output) = 0;

    //! Check whether energy output file is attached
    bool attached_energy_file()
    {
      if (energyfile_)
        return true;
      else
        return false;
    }

    //! Attach file handle for energy file #energyfile_
    virtual void attach_energy_file();

    //@}

    /*! @name Forces
     *
     *  Apply all sets of forces (external, internal, damping, inertia, ...)
     *  based on the current solution state.
     *
     *  On the level of Solid::TimInt, we only deal with forces. There are no
     *  stiffnesses since they are not needed in a general time integration
     *  scheme, but only in an implicit one.
     *
     *  For the application of forces AND
     *  stiffnesses, see Solid::TimIntImpl.
     *
     *  \sa Solid::TimIntImpl
     */
    //@{

    //! Apply external force
    void apply_force_external(const double time,                   //!< evaluation time
        const std::shared_ptr<Core::LinAlg::Vector<double>> dis,   //!< old displacement state
        const std::shared_ptr<Core::LinAlg::Vector<double>> disn,  //!< new displacement state
        Core::LinAlg::Vector<double>& vel,                         // velocity state
        Core::LinAlg::Vector<double>& fext                         //!< external force
    );

    /*! \brief Evaluate ordinary internal force
     *
     *  We need incremental displacements, because the internal variables,
     *  chiefly EAS parameters with an algebraic constraint, are treated
     *  as well. They are not treated perfectly, i.e. they are not iteratively
     *  equilibrated according to their (non-linear) constraint and
     *  the pre-determined displacements -- we talk explicit time integration
     *  here, but they are applied in linearised manner. The linearised
     *  manner means the static condensation is applied once with
     *  residual displacements replaced by the full-step displacement
     *  increment \f$D_{n+1}-D_{n}\f$.
     */
    void apply_force_internal(const double time,                   //!< evaluation time
        const double dt,                                           //!< step size
        std::shared_ptr<const Core::LinAlg::Vector<double>> dis,   //!< displacement state
        std::shared_ptr<const Core::LinAlg::Vector<double>> disi,  //!< incremental displacements
        const Core::LinAlg::Vector<double>& vel,                   // velocity state
        std::shared_ptr<Core::LinAlg::Vector<double>> fint         //!< internal force
    );

    //@}

    //! @name Nonlinear mass
    //@{

    //! Return bool indicating if we have nonlinear inertia forces
    Inpar::Solid::MassLin have_nonlinear_mass() const;

    //! check whether the initial conditions are fulfilled */
    virtual void nonlinear_mass_sanity_check(
        std::shared_ptr<const Core::LinAlg::Vector<double>> fext,  ///< external forces
        std::shared_ptr<const Core::LinAlg::Vector<double>> dis,   ///< displacements
        std::shared_ptr<const Core::LinAlg::Vector<double>> vel,   ///< velocities
        std::shared_ptr<const Core::LinAlg::Vector<double>> acc,   ///< accelerations
        const Teuchos::ParameterList* sdynparams = nullptr  ///< structural dynamics parameter list
    ) const;

    //@}

    //! Set forces due to interface with fluid, the force is expected external-force-like
    void set_force_interface(
        const Core::LinAlg::MultiVector<double>& iforce  ///< the force on interface
        ) override;

    //! @name Attributes
    //@{

    //! Provide Name
    virtual Inpar::Solid::DynamicType method_name() const = 0;

    //! Provide title
    std::string method_title() const;

    //! Return true, if time integrator is implicit
    virtual bool method_implicit() = 0;

    //! Return true, if time integrator is explicit
    bool method_explicit() { return (not method_implicit()); }

    //! Provide number of steps, e.g. a single-step method returns 1,
    //! a \f$m\f$-multistep method returns \f$m\f$
    virtual int method_steps() const = 0;

    //! return time integration factor
    double tim_int_param() const override = 0;

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

    //! @name Access methods
    //@{

    //! Access discretisation
    std::shared_ptr<Core::FE::Discretization> discretization() override
    {
      // Here a 'false' must be used. This is due to
      // the fact that TimInt possesses a references
      // on the discretisation #discret_ and not
      // a Teuchos::RefCountPointer. Eventually, TimInt
      // will be destroyed and it will immediately destroy
      // its #discret_ member. However, #discret_ is handed down
      // to the ConstrManager and kept there as a RefCountPointer.
      // The object #discret_ is gone, when ConstrManager tries
      // to kill it. We achieve a nice segmentation fault.
      // The 'false' prevents ConstrManager of trying to kill it.
      // return Teuchos::rcp(&discret_, false);

      // Now, the discretisation is stored as Teuchos::RefCountPointer,
      // thus
      return discret_;
    }

    //! Access to dofrowmap of discretization via const raw pointer
    const Core::LinAlg::Map* dof_row_map_view() override;

    //! Access solver, one of these have to be removed (see below)
    std::shared_ptr<Core::LinAlg::Solver> solver() { return solver_; }

    //! Access solver, one of these have to be removed (see above)
    std::shared_ptr<Core::LinAlg::Solver> linear_solver() override { return solver_; }

    //! Access solver for contact/meshtying problems
    std::shared_ptr<Core::LinAlg::Solver> contact_solver() { return contactsolver_; }

    //! Access output object
    std::shared_ptr<Core::IO::DiscretizationWriter> disc_writer() override { return output_; }

    //! Read restart values
    void read_restart(const int step  //!< restart step
        ) override;

    //! Set restart values
    void set_restart(int step,                               //!< restart step
        double time,                                         //!< restart time
        std::shared_ptr<Core::LinAlg::Vector<double>> disn,  //!< restart displacements
        std::shared_ptr<Core::LinAlg::Vector<double>> veln,  //!< restart velocities
        std::shared_ptr<Core::LinAlg::Vector<double>> accn,  //!< restart accelerations
        std::shared_ptr<std::vector<char>> elementdata,      //!< restart element data
        std::shared_ptr<std::vector<char>> nodedata          //!< restart element data
        ) override;

    //! Set the state of the nox group and the global state data container (implicit only)
    void set_state(const std::shared_ptr<Core::LinAlg::Vector<double>>& x) override
    {
      FOUR_C_THROW("new structural time integration only...");
    }

    //! Read and set restart state
    virtual void read_restart_state();

    //! Set restart state
    virtual void set_restart_state(
        std::shared_ptr<Core::LinAlg::Vector<double>> disn,  //!< restart displacements
        std::shared_ptr<Core::LinAlg::Vector<double>> veln,  //!< restart velocities
        std::shared_ptr<Core::LinAlg::Vector<double>> accn,  //!< restart accelerations
        std::shared_ptr<std::vector<char>> elementdata,      //!< restart element data
        std::shared_ptr<std::vector<char>> nodedata          //!< restart element data
    );

    //! Read and set restart forces
    virtual void read_restart_force() = 0;

    //! Read and set restart values for constraints
    void read_restart_constraint();

    //! Read and set restart values for Cardiovascular0D
    void read_restart_cardiovascular0_d();

    //! Read and set restart values for Spring Dashpot
    void read_restart_spring_dashpot();

    //! Read and set restart values for contact / meshtying
    void read_restart_contact_meshtying();

    //! initial guess of Newton's method
    std::shared_ptr<const Core::LinAlg::Vector<double>> initial_guess() override = 0;

    //! right-hand-side of Newton's method
    std::shared_ptr<const Core::LinAlg::Vector<double>> rhs() override = 0;

    /// set evaluation action
    void set_action_type(const Core::Elements::ActionType& action) override
    {
      FOUR_C_THROW("new structural time integration only...");
    }

    //! @name Access from outside via adapter (needed for coupled problems)
    //@{

    //! unknown displacements at \f$t_{n+1}\f$
    std::shared_ptr<const Core::LinAlg::Vector<double>> dispnp() const override { return disn_; }

    //! known displacements at \f$t_{n}\f$
    std::shared_ptr<const Core::LinAlg::Vector<double>> dispn() const override
    {
      return (*dis_)(0);
    }

    //! unknown velocity at \f$t_{n+1}\f$
    std::shared_ptr<const Core::LinAlg::Vector<double>> velnp() const override { return veln_; }

    //! unknown velocity at \f$t_{n}\f$
    std::shared_ptr<const Core::LinAlg::Vector<double>> veln() const override { return (*vel_)(0); }

    //! known velocity at \f$t_{n-1}\f$
    std::shared_ptr<const Core::LinAlg::Vector<double>> velnm() const override
    {
      return (*vel_)(-1);
    }

    //! unknown accelerations at \f$t_{n+1}\f$
    std::shared_ptr<const Core::LinAlg::Vector<double>> accnp() const override { return accn_; }

    //! known accelerations at \f$t_{n}\f$
    std::shared_ptr<const Core::LinAlg::Vector<double>> accn() const override { return (*acc_)(0); }

    //@}


    //! Access from inside of the structural time integrator
    //@{

    //! Return displacements \f$D_{n+1}\f$
    std::shared_ptr<Core::LinAlg::Vector<double>> dis_new() { return disn_; }

    //! Return displacements \f$D_{n}\f$
    std::shared_ptr<Core::LinAlg::Vector<double>> dis() { return (*dis_)(0); }

    //! Return velocities \f$V_{n+1}\f$
    std::shared_ptr<Core::LinAlg::Vector<double>> vel_new() { return veln_; }

    //! Return velocities \f$V_{n}\f$
    std::shared_ptr<Core::LinAlg::Vector<double>> vel() { return (*vel_)(0); }

    //! Return accelerations \f$A_{n+1}\f$
    std::shared_ptr<Core::LinAlg::Vector<double>> acc_new() { return accn_; }

    //! Return accelerations \f$A_{n}\f$
    std::shared_ptr<Core::LinAlg::Vector<double>> acc() { return (*acc_)(0); }

    //@}


    //! Return external force \f$F_{ext,n}\f$
    virtual std::shared_ptr<Core::LinAlg::Vector<double>> fext() = 0;

    //! Return external force \f$F_{ext,n+1}\f$
    virtual std::shared_ptr<Core::LinAlg::Vector<double>> fext_new() = 0;

    //! Return reaction forces
    std::shared_ptr<Core::LinAlg::Vector<double>> freact() override = 0;

    //! Return element data
    // std::shared_ptr<std::vector<char> > ElementData() {return discret_->PackMyElements();}

    //! dof map of vector of unknowns
    std::shared_ptr<const Core::LinAlg::Map> dof_row_map() override;

    //! dof map of vector of unknowns
    // new method for multiple dofsets
    std::shared_ptr<const Core::LinAlg::Map> dof_row_map(unsigned nds) override;

    //! Return stiffness,
    //! i.e. force residual differentiated by displacements
    std::shared_ptr<Core::LinAlg::SparseMatrix> system_matrix() override;

    //! Return stiffness,
    //! i.e. force residual differentiated by displacements
    std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> block_system_matrix() override;

    //! switch structure field to block matrix in fsi simulations
    void use_block_matrix(std::shared_ptr<const Core::LinAlg::MultiMapExtractor> domainmaps,
        std::shared_ptr<const Core::LinAlg::MultiMapExtractor> rangemaps) override = 0;

    //! Return sparse mass matrix
    std::shared_ptr<Core::LinAlg::SparseMatrix> mass_matrix();

    //! domain map of system matrix
    const Core::LinAlg::Map& domain_map() const override;

    //! are there any algebraic constraints?
    bool have_constraint() override = 0;

    //! are there any spring dashpot BCs?
    bool have_spring_dashpot() override = 0;

    //! get constraint manager defined in the structure
    std::shared_ptr<Constraints::ConstrManager> get_constraint_manager() override = 0;

    //! get spring dashpot manager defined in the structure
    std::shared_ptr<Constraints::SpringDashpotManager> get_spring_dashpot_manager() override = 0;

    //@}

    /// Access to scaling matrix for STC
    std::shared_ptr<Core::LinAlg::SparseMatrix> get_stc_mat() override
    {
      FOUR_C_THROW("STC is not implemented in the old time integration framework.");
    }

    //! @name Time step helpers
    //@{

    //! Return current time \f$t_{n}\f$
    double time_old() const override { return (*time_)[0]; }

    //! Return target time \f$t_{n+1}\f$
    double time() const override { return timen_; }

    //! Sets the current time \f$t_{n}\f$
    void set_time(const double time) override { (*time_)[0] = time; }

    //! Sets the target time \f$t_{n+1}\f$ of this time step
    void set_timen(const double time) override { timen_ = time; }

    //! Sets the current step \f$n+1\f$
    void set_step(int step) override { step_ = step; }

    //! Sets the current step \f$n+1\f$
    void set_stepn(int step) override { stepn_ = step; }

    //! Get upper limit of time range of interest
    double get_time_end() const override { return timemax_; }

    //! Set upper limit of time range of interest
    void set_time_end(double timemax) override { timemax_ = timemax; }

    //! Get time step size \f$\Delta t_n\f$
    double dt() const override { return (*dt_)[0]; }

    //! Set time step size \f$\Delta t_n\f$
    void set_dt(const double dtnew) override { (*dt_)[0] = dtnew; }

    //! Return current step number $n$
    int step_old() const override { return step_; }

    //! Return current step number $n+1$
    int step() const override { return stepn_; }

    //! Get number of time steps
    int num_step() const override { return stepmax_; }


    //! Return MapExtractor for Dirichlet boundary conditions
    std::shared_ptr<const Core::LinAlg::MapExtractor> get_dbc_map_extractor() override
    {
      return dbcmaps_;
    }

    //! Return (rotatory) transformation matrix of local co-ordinate systems
    std::shared_ptr<const Core::LinAlg::SparseMatrix> get_loc_sys_trafo() const;

    //! Return locsys manager
    std::shared_ptr<Core::Conditions::LocsysManager> locsys_manager() override
    {
      return locsysman_;
    }

    //@}

    //! @name Write access to field solution variables at \f$t^{n+1}\f$
    //@{

    /// write access to displacements at \f$t^{n+1}\f$
    std::shared_ptr<Core::LinAlg::Vector<double>> write_access_dispnp() override
    {
      return dis_new();
    }

    //! write access to velocities at \f$t_{n+1}\f$
    std::shared_ptr<Core::LinAlg::Vector<double>> write_access_velnp() override
    {
      return vel_new();
    }

    /// write access to displacements at \f$t^{n+1}\f$
    std::shared_ptr<Core::LinAlg::Vector<double>> write_access_dispn() override { return dis(); }

    //! write access to velocities at \f$t_{n+1}\f$
    std::shared_ptr<Core::LinAlg::Vector<double>> write_access_veln() override { return vel(); }

    //@}

    //! @name TSI specific methods
    //@{

    //! specific method for iterative staggered partitioned TSI

    /// Identify residual
    /// This method does not predict the target solution but
    /// evaluates the residual and the stiffness matrix.
    /// In partitioned solution schemes, it is better to keep the current
    /// solution instead of evaluating the initial guess (as the predictor)
    /// does.
    void prepare_partition_step() override = 0;

    //@}

    //! @name Contact and meshtying specific methods
    //@{

    //! return bool indicating if contact or meshtying are defined
    bool have_contact_meshtying() { return (cmtbridge_ != nullptr); }

    //! return contact/meshtying manager
    std::shared_ptr<CONTACT::MeshtyingContactBridge> meshtying_contact_bridge() override
    {
      return cmtbridge_;
    }

    /// do we have this model
    bool have_model(Inpar::Solid::ModelType model) override
    {
      FOUR_C_THROW("new structural time integration only");
      return false;
    }

    Solid::ModelEvaluator::Generic& model_evaluator(Inpar::Solid::ModelType mtype) override
    {
      FOUR_C_THROW("new time integration only");
    }

    /*!
    \brief Prepare time integration for contact/meshtying

    Check if contact / meshtying is chosen in input file. If yes, create manager object and
    initialize all relevant stuff.

    @param[in] sdynparams Structural dynamics input parameter list
    */
    void prepare_contact_meshtying(const Teuchos::ParameterList& sdynparams);

    /*!
    \brief Apply results of mesh initialization to the underlying problem discretization

    \note This is only necessary in case of a mortar method.

    \warning This routine modifies the reference coordinates of slave nodes at the meshtying
    interface.

    @param[in] Xslavemod Vector with modified nodal positions
    */
    void apply_mesh_initialization(std::shared_ptr<const Core::LinAlg::Vector<double>> Xslavemod);

    //! Prepare contact at the beginning of each new time step
    //! (call dynamic redistribution of contact interface(s) AND
    //! evaluate reference state for frictional contact at t=0)
    void prepare_step_contact();

    /// wrapper for things that should be done before prepare_time_step is called
    void pre_predict() final {};

    /// wrapper for things that should be done before solving the nonlinear iterations
    void pre_solve() final {};

    /// wrapper for things that should be done before updating
    void pre_update() final {};

    /// wrapper for things that should be done after solving the update
    void post_update() final {};

    /// wrapper for things that should be done after convergence of Newton scheme
    void post_output() final {};

    /// wrapper for things that should be done after the actual time loop is finished
    void post_time_loop() final {};

    //@}

    //! @name Biofilm methods
    //@{

    // reset everything (needed for biofilm simulations)
    void reset() override;

    // set structure displacement vector due to biofilm growth
    void set_str_gr_disp(std::shared_ptr<Core::LinAlg::Vector<double>> struct_growth_disp) override;

    virtual bool have_biofilm_growth() const { return strgrdisp_ != nullptr; }

    //@}

   protected:
    /// Expand the dbc map by dofs provided in Core::LinAlg::Map maptoadd
    void add_dirich_dofs(const std::shared_ptr<const Core::LinAlg::Map> maptoadd) override;

    /// Contract the dbc map by dofs provided in Core::LinAlg::Map maptoremove
    void remove_dirich_dofs(const std::shared_ptr<const Core::LinAlg::Map> maptoremove) override;

    //! @name General purpose algorithm members
    //@{
    std::shared_ptr<Core::FE::Discretization> discret_;  //!< attached discretisation

    int myrank_;  //!< ID of actual processor in parallel
    std::shared_ptr<Core::LinAlg::Solver>
        solver_;  //!< linear algebraic solver (no contact/meshtying)
    std::shared_ptr<Core::LinAlg::Solver>
        contactsolver_;           //!< linear algebraic solver (for contact/meshtying)
    bool solveradapttol_;         //!< adapt solver tolerance
    double solveradaptolbetter_;  //!< tolerance to which is adapted ????
    std::shared_ptr<Core::LinAlg::MapExtractor> dbcmaps_;  //!< map extractor object
                                                           //!< containing non-overlapping
                                                           //!< map of global DOFs on Dirichlet
                                                           //!< boundary conditions

    Inpar::Solid::DivContAct divcontype_;  //!< what to do when nonlinear solution fails
    int divconrefinementlevel_;  //!< number of refinement level in case of divercontype_ ==
                                 //!< adapt_step
    int divconnumfinestep_;      //!< number of converged time steps on current refinement level
                                 //!< in case of divercontype_ == adapt_step

    //! structural dynamic parameter list
    Teuchos::ParameterList sdynparams_;

    //@}

    //! @name Printing and output
    //@{
    std::shared_ptr<Core::IO::DiscretizationWriter> output_;  //!< binary output
    int printscreen_;                            //!< print infos to standard out every n steps
    bool printlogo_;                             //!< print the logo (or not)?
    bool printiter_;                             //!< print intermediate iterations during solution
    int outputcounter_;                          //!< output counter for OutputEveryIter
    int writerestartevery_;                      //!< write restart every given step;
                                                 //!< if 0, restart is not written
    bool writeele_;                              //!< write elements on/off
    bool writestate_;                            //!< write state on/off
    int writeresultsevery_;                      //!< write state/stress/strain every given step
    Inpar::Solid::StressType writestress_;       //!< stress output type
    Inpar::Solid::StrainType writestrain_;       //!< strain output type
    Inpar::Solid::StrainType writeplstrain_;     //!< plastic strain output type
    int writeenergyevery_;                       //!< write system energy every given step
    bool writesurfactant_;                       //!< write surfactant output
    bool writerotation_;                         //!< write strutural rotation tensor output
    std::shared_ptr<std::ofstream> energyfile_;  //!< outputfile for energy

    std::shared_ptr<std::vector<char>> stressdata_;    //!< container for element GP stresses
    std::shared_ptr<std::vector<char>> straindata_;    //!< container for element GP strains
    std::shared_ptr<std::vector<char>> plstraindata_;  //!< container for element GP plastic strains
    std::shared_ptr<std::vector<char>> rotdata_;       //!< container for element rotation tensor
    double kinergy_;                                   //!< kinetic energy
    double intergy_;                                   //!< internal energy
    double extergy_;                                   //!< external energy
    //@}

    //! @name Damping
    //!
    //! Rayleigh damping means \f${C} = c_\text{K} {K} + c_\text{M} {M}\f$
    //@{
    Inpar::Solid::DampKind damping_;  //!< damping type
    double dampk_;                    //!< damping factor for stiffness \f$c_\text{K}\f$
    double dampm_;                    //!< damping factor for mass \f$c_\text{M}\f$
    //@}

    //! @name Managed stuff
    //@{

    //! whatever constraints
    std::shared_ptr<Constraints::ConstrManager> conman_;      //!< constraint manager
    std::shared_ptr<Constraints::ConstraintSolver> consolv_;  //!< constraint solver

    // for 0D cardiovascular models
    std::shared_ptr<Utils::Cardiovascular0DManager> cardvasc0dman_;  //!< Cardiovascular0D manager

    // spring dashpot
    std::shared_ptr<Constraints::SpringDashpotManager> springman_;

    //! bridge for meshtying and contact
    std::shared_ptr<CONTACT::MeshtyingContactBridge> cmtbridge_;

    //! beam contact
    std::shared_ptr<CONTACT::Beam3cmanager> beamcman_;

    //! Dirichlet BCs with local co-ordinate system
    std::shared_ptr<Core::Conditions::LocsysManager> locsysman_;

    //! Map to differentiate pressure and displacement/velocity DOFs
    std::shared_ptr<Core::LinAlg::MapExtractor> pressure_;

    //! Is GMSH output of displacements required?
    bool gmsh_out_;

    //@}

    //! @name General control parameters
    //@{
    std::shared_ptr<TimeStepping::TimIntMStep<double>>
        time_;      //!< time \f$t_{n}\f$ of last converged step
    double timen_;  //!< target time \f$t_{n+1}\f$
    std::shared_ptr<TimeStepping::TimIntMStep<double>> dt_;  //!< time step size \f$\Delta t\f$
    double timemax_;                                         //!< final time \f$t_\text{fin}\f$
    int stepmax_;                                            //!< final step \f$N\f$
    int step_;                                               //!< time step index \f$n\f$
    int stepn_;                                              //!< time step index \f$n+1\f$
    double rand_tsfac_;      //!< random factor for modifying time-step size in case this way of
                             //!< continuing non-linear iteration was chosen
    bool firstoutputofrun_;  //!< flag whether this output step is the first one (restarted or not)
    bool lumpmass_;          //!< flag for lumping the mass matrix, default: false
    //@}

    //! @name Global vectors
    //@{
    std::shared_ptr<Core::LinAlg::Vector<double>> zeros_;  //!< a zero vector of full length
    //@}

    //! @name Global state vectors
    //@{

    //! global displacements \f${D}_{n}, D_{n-1}, ...\f$
    std::shared_ptr<TimeStepping::TimIntMStep<Core::LinAlg::Vector<double>>> dis_;

    //! global velocities \f${V}_{n}, V_{n-1}, ...\f$
    std::shared_ptr<TimeStepping::TimIntMStep<Core::LinAlg::Vector<double>>> vel_;

    //! global accelerations \f${A}_{n}, A_{n-1}, ...\f$
    std::shared_ptr<TimeStepping::TimIntMStep<Core::LinAlg::Vector<double>>> acc_;

    //!< global displacements \f${D}_{n+1}\f$ at \f$t_{n+1}\f$
    std::shared_ptr<Core::LinAlg::Vector<double>> disn_;

    //!< global velocities \f${V}_{n+1}\f$ at \f$t_{n+1}\f$
    std::shared_ptr<Core::LinAlg::Vector<double>> veln_;

    //!< global accelerations \f${A}_{n+1}\f$ at \f$t_{n+1}\f$
    std::shared_ptr<Core::LinAlg::Vector<double>> accn_;

    //!< global internal force
    std::shared_ptr<Core::LinAlg::Vector<double>> fint_;

    //! additional external forces (e.g. interface force in FSI)
    std::shared_ptr<Core::LinAlg::Vector<double>> fifc_;

    //!< pure structural global internal force, i.e. no condensation of EAS, plasticity,...
    std::shared_ptr<Core::LinAlg::Vector<double>> fresn_str_;

    //!< pure structural global internal force at \f$t_n\f$, i.e. no condensation of EAS,
    //!< plasticity,...
    std::shared_ptr<Core::LinAlg::Vector<double>> fintn_str_;

    //@}


    //! @name System matrices
    //@{
    //! holds eventually effective stiffness
    std::shared_ptr<Core::LinAlg::SparseOperator> stiff_;

    //! mass matrix (constant)
    std::shared_ptr<Core::LinAlg::SparseOperator> mass_;

    //! damping matrix
    std::shared_ptr<Core::LinAlg::SparseOperator> damp_;
    //@}

    //! @name Time measurement
    //@{
    std::shared_ptr<Teuchos::Time> timer_;  //!< timer for solution technique
    double dtsolve_;                        //!< linear solver time
    double dtele_;                          //!< element evaluation time
    double dtcmt_;                          //!< contact / meshtying evaluation time
    double inttime_global_;                 //!< global integration time for contact evaluation
    //@}

    //! @name Biofilm specific stuff
    //@{
    std::shared_ptr<Core::LinAlg::Vector<double>> strgrdisp_;
    //@}

    //! @name porous media specific stuff
    //@{
    std::shared_ptr<Core::LinAlg::MapExtractor> porositysplitter_;
    //@}

    std::shared_ptr<Cardiovascular0D::ProperOrthogonalDecomposition>
        mor_;  //!< model order reduction

   private:
    //! flag indicating if class is setup
    bool issetup_;

    //! flag indicating if class is initialized
    bool isinit_;

   protected:
    //! returns true if setup() was called and is still valid
    bool is_setup() { return issetup_; };

    //! returns true if init(..) was called and is still valid
    bool is_init() const { return isinit_; };

    //! check if \ref setup() was called
    void check_is_setup()
    {
      if (not is_setup()) FOUR_C_THROW("setup() was not called.");
    };

    //! check if \ref init() was called
    void check_is_init() const
    {
      if (not is_init()) FOUR_C_THROW("init(...) was not called.");
    };

   public:
    //! set flag true after setup or false if setup became invalid
    void set_is_setup(bool trueorfalse) { issetup_ = trueorfalse; };

    //! set flag true after init or false if init became invalid
    void set_is_init(bool trueorfalse) { isinit_ = trueorfalse; };

  };  // class TimInt

}  // namespace Solid

/*----------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
