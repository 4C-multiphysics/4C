// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SCATRA_TIMINT_IMPLICIT_HPP
#define FOUR_C_SCATRA_TIMINT_IMPLICIT_HPP

#include "4C_config.hpp"

#include "4C_adapter_scatra_wrapper.hpp"
#include "4C_fem_condition.hpp"
#include "4C_inpar_fluid.hpp"
#include "4C_inpar_scatra.hpp"
#include "4C_io_discretization_visualization_writer_mesh.hpp"
#include "4C_io_runtime_csv_writer.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_utils_result_test.hpp"

#include <iostream>
#include <memory>
#include <optional>
#include <set>
#include <utility>

FOUR_C_NAMESPACE_OPEN

/*==========================================================================*/
// forward declarations
/*==========================================================================*/
namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Global
{
  class Problem;
}

namespace Core::IO
{
  class DiscretizationReader;
  class DiscretizationWriter;
  class InputControl;
  class RuntimeCsvWriter;
}  // namespace Core::IO

namespace Core::LinAlg
{
  class Solver;
  class SparseMatrix;
  class MapExtractor;
  class MultiMapExtractor;
  class BlockSparseMatrixBase;
  class SparseOperator;
  class KrylovProjector;
  enum class EquilibrationMethod;
  enum class MatrixType;
}  // namespace Core::LinAlg

namespace Core::DOFSets
{
  class DofSet;
}  // namespace Core::DOFSets


namespace FLD
{
  class DynSmagFilter;
  class Vreman;
}  // namespace FLD

// forward declaration
namespace CONTACT
{
  class NitscheStrategySsi;
}  // namespace CONTACT

namespace ScaTra
{
  class HomoIsoTurbScalarForcing;
  class MeshtyingStrategyBase;
  class ScalarHandler;
  class OutputScalarsStrategyBase;
  class OutputScalarsStrategyDomain;
  class OutputScalarsStrategyCondition;
  class OutputDomainIntegralStrategy;

  /*!
   * \brief implicit time integration for scalar transport problems
   */

  class ScaTraTimIntImpl : public Adapter::ScatraInterface
  {
    friend class HomoIsoTurbInitialScalarField;
    friend class HomoIsoTurbScalarForcing;
    friend class OutputScalarsStrategyBase;
    friend class OutputScalarsStrategyDomain;
    friend class OutputScalarsStrategyCondition;

   public:
    /*========================================================================*/
    //! @name Constructors and destructors and related methods
    /*========================================================================*/

    //! Standard Constructor
    ScaTraTimIntImpl(std::shared_ptr<Core::FE::Discretization> actdis,  //!< discretization
        std::shared_ptr<Core::LinAlg::Solver> solver,                   //!< linear solver
        std::shared_ptr<Teuchos::ParameterList> params,                 //!< parameter list
        std::shared_ptr<Teuchos::ParameterList> extraparams,     //!< supplementary parameter list
        std::shared_ptr<Core::IO::DiscretizationWriter> output,  //!< output writer
        int probnum = 0                                          //!< global problem number
    );

    //! don't want copy constructor
    ScaTraTimIntImpl(const ScaTraTimIntImpl& old) = delete;

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
    \return void

    */
    virtual void init();

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
    virtual void setup();

    //! setup the context vector that defines the names for the output of the solution vector
    //! #phinp_
    virtual void setup_context_vector();

    //! Initialization of turbulence models
    void init_turbulence_model(
        const Core::LinAlg::Map* dofrowmap, const Core::LinAlg::Map* noderowmap);

    /*========================================================================*/
    //! @name general framework
    /*========================================================================*/

    /*--- set, prepare, and predict ------------------------------------------*/

    //! add global state vectors specific for time-integration scheme
    void add_time_integration_specific_vectors(bool forcedincrementalsolver = false) override;

    //! initialize system matrix
    [[nodiscard]] std::shared_ptr<Core::LinAlg::SparseOperator> init_system_matrix() const;

    //! prepare time loop
    virtual void prepare_time_loop();

    //! setup the variables to do a new time step
    virtual void prepare_time_step();

    //! initialization procedure prior to evaluation of first time step
    virtual void prepare_first_time_step();

    //! preparations for solve
    virtual void prepare_linear_solve();

    //! set time to @p newtime and step value to @p newstep
    void set_time_step(const double newtime, const int newstep)
    {
      time_ = newtime;
      step_ = newstep;
    }

    //! set time stepping information from this time integration to micro scales
    void set_time_stepping_to_micro_scale();

    //! set timestep value
    virtual void set_dt(const double newdt)
    {
      dta_ = newdt;
      // We have to set the newdt in the ele_calc's as well:
      set_element_time_parameter();
    }

    //! do explicit predictor step to obtain better starting value for Newton-Raphson iteration
    virtual void explicit_predictor() const;

    //! set the velocity field (zero or field by function)
    void set_velocity_field_from_function();

    /*! Set external force field

     The contribution of velocity due to the external force \f$ v_{external force} \f$ to the
     convection-diffusion-reaction equation is given by
     \f[
        \nabla \cdot (v_{external force} \phi).
     \f]
     The velocity due to the external force F is given by
     \f[
        v_{external force} = M \cdot F,
     \f]
     where M is the intrinsic mobility of the scalar.
     */
    void set_external_force() const;

    //! set the @acceleration vector to the scalar transport discretization
    void set_acceleration_field(const Core::LinAlg::Vector<double>& acceleration) const;

    //! set the @convective_velocity vector to the scalar transport discretization
    void set_convective_velocity(const Core::LinAlg::Vector<double>& convective_velocity) const;

    //! set the @fine_scale_velocity vector to the scalar transport discretization
    void set_fine_scale_velocity(const Core::LinAlg::Vector<double>& fine_scale_velocity) const;

    //! return whether setting of the fine scale velocity is required
    [[nodiscard]] bool fine_scale_velocity_field_required() const;

    //! set the @velocity vector to the scalar transport discretization
    void set_velocity_field(const Core::LinAlg::Vector<double>& velocity);

    //! set the @wall_shear_stress vector to the scalar transport discretization
    void set_wall_shear_stresses(const Core::LinAlg::Vector<double>& wall_shear_stress) const;

    //! set the @pressure vector to the scalar transport discretization
    void set_pressure_field(const Core::LinAlg::Vector<double>& pressure) const;

    void set_membrane_concentration(
        std::shared_ptr<const Core::LinAlg::Vector<double>> MembraneConc);

    void set_mean_concentration(std::shared_ptr<const Core::LinAlg::Vector<double>> MeanConc);

    void clear_external_concentrations()
    {
      mean_conc_ = nullptr;
      membrane_conc_ = nullptr;
    }

    //! read restart data
    virtual void read_restart(int step, std::shared_ptr<Core::IO::InputControl> input = nullptr);

    //! setup natural convection
    virtual void setup_nat_conv();

    //! set number of dofset to write displacement values on
    void set_number_of_dof_set_displacement(int nds_disp)
    {
      FOUR_C_ASSERT(nds_disp_ == -1, "Don't set 'nds_disp_' twice!");
      nds_disp_ = nds_disp;
    }

    //! set number of dofset to write interface growth values on
    void set_number_of_dof_set_growth(int nds_growth)
    {
      FOUR_C_ASSERT(nds_growth_ == -1, "Don't set 'nds_growth_' twice!");
      nds_growth_ = nds_growth;
    }

    //! set number of dofset to write micro scale values on
    void set_number_of_dof_set_micro_scale(int nds_micro)
    {
      FOUR_C_ASSERT(nds_micro_ == -1, "Don't set 'nds_micro_' twice!");
      nds_micro_ = nds_micro;
    }

    //! set number of dofset to write pressure values on
    void set_number_of_dof_set_pressure(int nds_pressure)
    {
      FOUR_C_ASSERT(nds_pres_ == -1, "Don't set 'nds_pres_' twice!");
      nds_pres_ = nds_pressure;
    }

    //! set number of dofset to write scalar transport values on
    void set_number_of_dof_set_scatra(int nds_scatra)
    {
      FOUR_C_ASSERT(nds_scatra_ == -1, "Don't set 'nds_scatra_' twice!");
      nds_scatra_ = nds_scatra;
    }

    //! set number of dofset to write thermo values on
    void set_number_of_dof_set_thermo(int nds_thermo)
    {
      FOUR_C_ASSERT(nds_thermo_ == -1, "Don't set 'nds_thermo_' twice!");
      nds_thermo_ = nds_thermo;
    }

    //! set number of dofset to write two-tensor quantities on, e.g. stresses, strains
    void set_number_of_dof_set_two_tensor_quantity(int nds_two_tensor_quantity)
    {
      FOUR_C_ASSERT(nds_two_tensor_quantity_ == -1, "Don't set 'nds_two_tensor_quantity_' twice!");
      nds_two_tensor_quantity_ = nds_two_tensor_quantity;
    }

    //! set number of dofset to write velocity values on
    void set_number_of_dof_set_velocity(int nds_velocity)
    {
      FOUR_C_ASSERT(nds_vel_ == -1, "Don't set 'nds_vel_' twice!");
      nds_vel_ = nds_velocity;
    }

    //! set number of dofset to write wall shear stress values on
    void set_number_of_dof_set_wall_shear_stress(int nds_wall_shear_stress)
    {
      FOUR_C_ASSERT(nds_wss_ == -1, "Don't set 'nds_wss_' twice!");
      nds_wss_ = nds_wall_shear_stress;
    }

    //! returns the maximum dofset number that is set
    [[nodiscard]] int get_max_dof_set_number() const;

    //! store reaction coefficient for macro-micro coupling with deforming macro dis
    void set_macro_micro_rea_coeff(const double macro_micro_rea_coeff)
    {
      macro_micro_rea_coeff_ = macro_micro_rea_coeff;
    }

    //! set the Nitsche contact strategy that contributes to the RHS
    void set_nitsche_contact_strategy(std::shared_ptr<CONTACT::NitscheStrategySsi> strategy_ptr)
    {
      contact_strategy_nitsche_ = std::move(strategy_ptr);
    }

    //! create result test for scalar transport field
    virtual std::shared_ptr<Core::Utils::ResultTest> create_scatra_field_test();

    //! Add tests to global problem and start tests
    virtual void test_results();

    /*--- calculate and update -----------------------------------------------*/

    //! do time integration (time loop)
    virtual void time_loop();

    //! operator for manipulations before call to \ref Solve() ; May be overridden by subclass.
    virtual void pre_solve() {};

    //! general solver call for coupled algorithms (decides if linear/nonlinear internally)
    virtual void solve();

    //! operator for manipulations after call to \ref Solve() ; May be overridden by subclass.
    virtual void post_solve() {};

    //! update solution after convergence of the nonlinear Newton-Raphson iteration
    virtual void update();

    /*!
     * @brief apply moving mesh data
     *
     * @param[in] dispnp  displacement vector
     */
    void apply_mesh_movement(const Core::LinAlg::Vector<double>& dispnp) const;

    //! calculate fluxes inside domain and/or on boundary and write result to file if
    //! @p writetofile is true
    void calc_flux(bool writetofile);

    //! calculate flux vector field inside computational domain
    [[nodiscard]] std::shared_ptr<Core::LinAlg::MultiVector<double>> calc_flux_in_domain() const;

    //! calculate mass/heat normal flux at specified boundaries and write result to file if @p
    //! writetofile is true
    std::shared_ptr<Core::LinAlg::MultiVector<double>> calc_flux_at_boundary(bool writetofile);

    //! calculation of relative error with reference to analytical solution
    virtual void evaluate_error_compared_to_analytical_sol();

    //! finite difference check for scalar transport system matrix
    virtual void fd_check();

    //! apply Neumann and Dirichlet BC to system
    void apply_bc_to_system();

    void evaluate_initial_time_derivative(std::shared_ptr<Core::LinAlg::SparseOperator> matrix,
        std::shared_ptr<Core::LinAlg::Vector<double>> rhs);

    //! prepare time integrator specific things before calculation of initial time derivative
    virtual void pre_calc_initial_time_derivative() {}

    //! clean up settings from pre_calc_initial_time_derivative() after the initial time derivative
    //! is calculated
    virtual void post_calc_initial_time_derivative() {}

    //! calculate mean concentrations of micro discretization at nodes
    void calc_mean_micro_concentration();

    /*--- query and output ---------------------------------------------------*/

    //! return ALE flag
    [[nodiscard]] bool is_ale() const { return isale_; }

    //! return flag for macro scale in multi-scale simulations
    [[nodiscard]] bool macro_scale() const { return macro_scale_; }

    //! return the type of equilibration for the global system of scalar transport equations
    [[nodiscard]] Core::LinAlg::EquilibrationMethod equilibration_method() const
    {
      return equilibrationmethod_;
    }

    //! return the type of global system matrix in global system of equations
    [[nodiscard]] Core::LinAlg::MatrixType matrix_type() const { return matrixtype_; }

    //! Provide enum of the time integration scheme
    [[nodiscard]] Inpar::ScaTra::TimeIntegrationScheme method_name() const { return timealgo_; }

    //! Provide title of the time integration scheme
    std::string method_title() { return map_tim_int_enum_to_string(method_name()); }

    //! return flag for micro scale in multi-scale simulations
    [[nodiscard]] bool micro_scale() const { return micro_scale_; };

    //! print information about the current time step to screen
    virtual void print_time_step_info();

    //! return system matrix as a sparse operator
    std::shared_ptr<Core::LinAlg::SparseOperator> system_matrix_operator() { return sysmat_; };

    //! return system matrix cast to sparse matrix
    std::shared_ptr<Core::LinAlg::SparseMatrix> system_matrix();

    //! return system matrix cast to block sparse matrix
    std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> block_system_matrix();

    //! return the map extractor associated with blocks of global system matrix
    [[nodiscard]] std::shared_ptr<const Core::LinAlg::MultiMapExtractor> dof_block_maps() const
    {
      return dof_block_maps_;
    }

    //! return the map extractor associated with the nodes inside the blocks of global system matrix
    [[nodiscard]] std::shared_ptr<const Core::LinAlg::MultiMapExtractor> node_block_maps() const
    {
      return node_block_maps_;
    }

    //! return residual vector
    [[nodiscard]] std::shared_ptr<Core::LinAlg::Vector<double>> residual() const
    {
      return residual_;
    }

    //! return trueresidual vector
    std::shared_ptr<Core::LinAlg::Vector<double>> true_residual() { return trueresidual_; }

    //! return increment vector
    [[nodiscard]] std::shared_ptr<Core::LinAlg::Vector<double>> increment() const
    {
      return increment_;
    }

    //! return flag indicating if an incremental solution approach is used
    [[nodiscard]] bool is_incremental() const { return incremental_; }

    //! return Krylov projector
    std::shared_ptr<Core::LinAlg::KrylovProjector> projector() { return projector_; }

    //! return number of dofset associated with displacement dofs
    [[nodiscard]] int nds_disp() const override { return nds_disp_; }

    //! return number of dofset associated with interface growth dofs
    [[nodiscard]] int nds_growth() const { return nds_growth_; }

    //! return number of dofset to store nodal micro quantities on macro discretisation
    [[nodiscard]] int nds_micro() const { return nds_micro_; }

    //! return number of dofset associated with pressure dofs
    [[nodiscard]] int nds_pressure() const { return nds_pres_; }

    //! return number of dofset associated with scalar transport dofs
    [[nodiscard]] int nds_scatra() const { return nds_scatra_; }

    //! return number of dofset associated with thermo dofs
    [[nodiscard]] int nds_thermo() const { return nds_thermo_; }

    //! return number of dofset associated with two-tensor quantity dofs, e.g. stresses, strains
    [[nodiscard]] int nds_two_tensor_quantity() const { return nds_two_tensor_quantity_; }

    //! return number of dofset associated with velocity dofs
    [[nodiscard]] int nds_vel() const { return nds_vel_; }

    //! return number of dofset associated with wall shear stress dofs
    [[nodiscard]] int nds_wall_shear_stress() const { return nds_wss_; }

    //! return domain flux vector
    [[nodiscard]] std::shared_ptr<const Core::LinAlg::MultiVector<double>> flux_domain() const
    {
      return flux_domain_;
    }

    //! return boundary flux vector
    [[nodiscard]] std::shared_ptr<const Core::LinAlg::MultiVector<double>> flux_boundary() const
    {
      return flux_boundary_;
    }

    //! return Dirichlet map
    std::shared_ptr<const Core::LinAlg::MapExtractor> dirich_maps() { return dbcmaps_; }

    //! add dirichlet dofs to dbcmaps_
    void add_dirich_cond(const std::shared_ptr<const Core::LinAlg::Map> maptoadd);

    //! remove dirichlet dofs from dbcmaps_
    void remove_dirich_cond(const std::shared_ptr<const Core::LinAlg::Map> maptoremove);

    //! return pointer to const dofrowmap
    std::shared_ptr<const Core::LinAlg::Map> dof_row_map();

    //! return pointer to const dofrowmap of specified dofset
    std::shared_ptr<const Core::LinAlg::Map> dof_row_map(int nds);

    //! return discretization
    [[nodiscard]] std::shared_ptr<Core::FE::Discretization> discretization() const override
    {
      return discret_;
    }

    //! return the parameter lists
    [[nodiscard]] std::shared_ptr<Teuchos::ParameterList> scatra_parameter_list() const
    {
      return params_;
    }
    [[nodiscard]] std::shared_ptr<Teuchos::ParameterList> scatra_extra_parameter_list() const
    {
      return extraparams_;
    }
    virtual std::shared_ptr<Teuchos::ParameterList> scatra_time_parameter_list() = 0;

    //! Access output object: CD-Rom and DVD only - no BlueRay support!!! ;)
    [[nodiscard]] std::shared_ptr<Core::IO::DiscretizationWriter> disc_writer() const
    {
      return output_;
    }

    //! returns the map extractor used for convergence check either in ELCH or LOMA case
    [[nodiscard]] std::shared_ptr<Core::LinAlg::MapExtractor> splitter() const { return splitter_; }

    //! Checks if output of results or restart information is required and writes data to disk
    virtual void check_and_write_output_and_restart();

    //! write restart data to disk
    virtual void write_restart() const;

    //! write results to disk
    void write_result();

    //! collect runtime output data
    virtual void collect_runtime_output_data();

    //! collect the runtime output data and write it to disk
    void write_runtime_output();

    //! Convergence check for two-way coupled ScaTra problems.
    [[nodiscard]] bool convergence_check(int itnum, int itmax, double ittol) const;

    //! return solver
    [[nodiscard]] std::shared_ptr<Core::LinAlg::Solver> solver() const { return solver_; }

    //! return parameters for finite difference check
    [[nodiscard]] Inpar::ScaTra::FdCheck fd_check_type() const { return fdcheck_; }
    [[nodiscard]] double fd_check_eps() const { return fdcheckeps_; }
    [[nodiscard]] double fd_check_tol() const { return fdchecktol_; }

    //! return meshtying strategy (includes standard case without meshtying)
    [[nodiscard]] std::shared_ptr<ScaTra::MeshtyingStrategyBase> strategy() const override
    {
      return strategy_;
    }

    //! return flag indicating the availability of scatra-scatra interface kinetics condition(s)
    [[nodiscard]] bool s2i_kinetics() const { return s2ikinetics_; }

    //! return flag for scatra-scatra interface mesh tying
    [[nodiscard]] bool s2i_meshtying() const { return s2imeshtying_; }

    //! return relative errors of scalar fields in L2 and H1 norms
    [[nodiscard]] std::shared_ptr<std::vector<double>> rel_errors() const { return relerrors_; }

    //! output performance statistics associated with linear solver into *.csv file
    static void output_lin_solver_stats(const Core::LinAlg::Solver& solver,  //!< linear solver
        const double& time,    //!< solver time maximized over all processors
        const int& step,       //!< time step
        const int& iteration,  //!< Newton-Raphson iteration number
        const int& size        //!< size of the linear system
    );

    //! output performance statistics associated with nonlinear solver into *.csv file
    static void output_nonlin_solver_stats(
        const int& iterations,  //!< iteration count of nonlinear solver
        const double& time,     //!< solver time maximized over all processors
        const int& step,        //!< time step
        MPI_Comm comm           //!< communicator
    );

    /*========================================================================*/
    //! @name Time, time-step and related methods
    /*========================================================================*/

    /*--- calculate and update -----------------------------------------------*/

    //! determine whether there are still time steps to be evaluated
    [[nodiscard]] virtual bool not_finished() const
    {
      return step_ < stepmax_ and time_ + 1.e-12 < maxtime_;
    }

    /*--- query and output ---------------------------------------------------*/

    //! return current time value
    [[nodiscard]] double time() const { return time_; }

    //! return the current step number
    [[nodiscard]] int step() const { return step_; }

    //! total number of time steps ? rename StepMax?
    [[nodiscard]] int n_step() const { return stepmax_; }

    //! return number of newton iterations in last timestep
    [[nodiscard]] int iter_num() const { return iternum_; }

    //! return number of outer iterations in partitioned simulations
    [[nodiscard]] unsigned iter_num_outer() const { return iternum_outer_; }

    //! return time step size
    [[nodiscard]] double dt() const { return dta_; }

    //! return if the time step was changed during adapt_time_step_size()
    [[nodiscard]] bool time_step_adapted() const { return timestepadapted_; }

    /*========================================================================*/
    //! @name scalar degrees of freedom and related
    /*========================================================================*/

    /*--- set, prepare, and predict ------------------------------------------*/

    //! set the initial scalar field phi
    virtual void set_initial_field(Inpar::ScaTra::InitialField init,  //!< type of initial field
        int startfuncno  //!< number of the space-time function
    );

    /*========================================================================*/
    //! @name Preconditioning
    /*========================================================================*/

    virtual void setup_splitter() {}

    //! set up the (block) maps of the scatra system matrix and the meshtying object
    void setup_matrix_block_maps_and_meshtying();

    //! set up the (block) maps of the scatra system matrix
    void setup_matrix_block_maps();

    //! some of the set up of the (block) maps of the scatra system matrix has to be done after
    //! setup_meshtying() has been called
    void post_setup_matrix_block_maps() const;

    /*!
     * @brief build maps associated with blocks of global system matrix
     *
     * @param[in]  partitioning_conditions domain partitioning conditions
     * @param[out] dof_block_maps          vector for degrees of freedom maps for each matrix block
     *                                     to be built
     * @param[out] node_block_maps         vector for node maps for each matrix block to be built
     */
    virtual void build_block_maps(
        const std::vector<const Core::Conditions::Condition*>& partitioning_conditions,
        std::vector<std::shared_ptr<const Core::LinAlg::Map>>& dof_block_maps,
        std::vector<std::shared_ptr<const Core::LinAlg::Map>>& node_block_maps) const;

    //! Build null spaces associated with blocks of global system matrix. Hand in solver to access
    //! the parameter list and initial number of the block (e.g. for coupled problems)
    void build_block_null_spaces(const Core::LinAlg::Solver& solver, int init_block_number) const;

    /*--- calculate and update -----------------------------------------------*/

    //! call elements to calculate system matrix and rhs and assemble
    virtual void assemble_mat_and_rhs();

    //! compute time derivatives of discrete state variables
    virtual void compute_time_derivative();

    //! compute parameters of the Input voltage to use for the double layer current density
    virtual void compute_time_deriv_pot0(bool init) = 0;

    //! compute values at intermediate time steps (required for generalized-alpha) ? rename?
    virtual void compute_intermediate_values() = 0;

    //! compute values at the interior of the elements (required for hdg)
    virtual void compute_interior_values() = 0;

    //! compute nodal density values from nodal concentration values (natural convection)
    void compute_density();

    //! evaluate macro-micro coupling on micro scale in multi-scale scalar transport problems
    void evaluate_macro_micro_coupling();

    //! iterative update of phinp
    void update_iter(const Core::LinAlg::Vector<double>& inc  //!< increment vector for phi
    );

    /*--- query and output ---------------------------------------------------*/

    //! return maximum number of dofs per node
    [[nodiscard]] int max_num_dof_per_node() const;

    //! return number of transported scalars
    [[nodiscard]] int num_scal() const;

    //! return number of dofs per node
    [[nodiscard]] int num_dof_per_node() const;

    //! return the number of dofs per node in the @p condition
    [[nodiscard]] int num_dof_per_node_in_condition(
        const Core::Conditions::Condition& condition) const;

    //! return the number of transported scalars per node in the @p condition
    [[nodiscard]] virtual int num_scal_in_condition(
        const Core::Conditions::Condition& condition) const
    {
      return num_dof_per_node_in_condition(condition);
    }

    //! return relaxation parameters
    std::vector<double>& omega() { return omega_; }

    //! return relaxation parameters
    [[nodiscard]] const std::vector<double>& omega() const { return omega_; }

    //! return scalar field phi at time n
    std::shared_ptr<Core::LinAlg::Vector<double>> phin() override { return phin_; }

    //! return scalar field phi at time n+1
    [[nodiscard]] std::shared_ptr<Core::LinAlg::Vector<double>> phinp() const { return phinp_; }

    //! get mean concentration of micro discretization
    [[nodiscard]] std::shared_ptr<Core::LinAlg::Vector<double>> phinp_micro() const
    {
      return phinp_micro_;
    }

    //! return increment of scalar field phi at time n+1 for partitioned simulations
    [[nodiscard]] std::shared_ptr<Core::LinAlg::Vector<double>> phinp_inc() const
    {
      return phinp_inc_;
    }

    //! set increment of scalar field phi at time n+1 for partitioned simulations
    void set_phinp_inc(std::shared_ptr<Core::LinAlg::Vector<double>> phinp_inc)
    {
      phinp_inc_ = std::move(phinp_inc);
    }

    //! return increment of scalar field phi at time n+1 from the previous outer iteration step for
    //! partitioned simulations
    [[nodiscard]] std::shared_ptr<Core::LinAlg::Vector<double>> phinp_inc_old() const
    {
      return phinp_inc_old_;
    }

    //! set increment of scalar field phi at time n+1 from the previous outer iteration step for
    //! partitioned simulations
    void set_phinp_inc_old(std::shared_ptr<Core::LinAlg::Vector<double>> phinp_inc_old)
    {
      phinp_inc_old_ = std::move(phinp_inc_old);
    }

    //! return time derivative of scalar field phi at time n
    [[nodiscard]] std::shared_ptr<Core::LinAlg::Vector<double>> phidtn() const { return phidtn_; }

    //! return time derivative of scalar field phi at time n+1
    [[nodiscard]] std::shared_ptr<Core::LinAlg::Vector<double>> phidtnp() const { return phidtnp_; }

    //! return scalar field history
    [[nodiscard]] std::shared_ptr<Core::LinAlg::Vector<double>> hist() const { return hist_; }

    //! return scalar field phi at time n+alpha_F
    virtual std::shared_ptr<Core::LinAlg::Vector<double>> phiaf() = 0;

    //! return scalar field phi at time n+alpha_F (gen-alpha) or n+1 (otherwise)
    virtual std::shared_ptr<Core::LinAlg::Vector<double>> phiafnp() { return phinp_; }

    //! return scalar field phi at time n+alpha_M
    virtual std::shared_ptr<Core::LinAlg::Vector<double>> phiam() = 0;

    //! return time derivative of scalar field phi at time n+alpha_M
    virtual std::shared_ptr<Core::LinAlg::Vector<double>> phidtam() = 0;

    //! return fine-scale scalar field fsphi at time n+1 or alpha_M
    virtual std::shared_ptr<Core::LinAlg::Vector<double>> fs_phi() = 0;

    //! output total and mean values of transported scalars
    virtual void output_total_and_mean_scalars(int num = 0);

    //! output domain or boundary integrals, i.e., surface areas or volumes of specified nodesets
    void output_domain_or_boundary_integrals(const std::string& condstring);

    //! output of reaction(s) integral
    void output_integr_reac(int num = 0) const;

    //! return density field at time n+alpha_F (gen-alpha) or n+1 (otherwise) for natural convection
    std::shared_ptr<Core::LinAlg::Vector<double>> densafnp() { return densafnp_; }

    //! problem-specific restart
    virtual void read_restart_problem_specific(
        const int step, Core::IO::DiscretizationReader& reader) {};

    //! return time for evaluation of elements
    [[nodiscard]] double dt_ele() const { return dtele_; }

    //! return the time for the solution of the linear system of equations
    [[nodiscard]] double dt_solve() const { return dtsolve_; }

    //! return total values of transported scalars
    [[nodiscard]] const std::map<const int, std::vector<double>>& total_scalars() const;

    //! return mean values of transported scalars
    [[nodiscard]] const std::map<const int, std::vector<double>>& mean_scalars() const;

    //! return values of domain integrals
    [[nodiscard]] const std::vector<double>& domain_integrals() const;

    //! return values of boundary integrals
    [[nodiscard]] const std::vector<double>& boundary_integrals() const;

    //! return micro-scale coupling flux for macro-micro coupling in multi-scale simulations
    [[nodiscard]] double q() const { return q_; }

    //! derivative of micro-scale coupling flux w.r.t. macro-scale state variable for macro-micro
    //! coupling in multi-scale simulations
    [[nodiscard]] const std::vector<double>& dq_dphi() const { return dq_dphi_; }

    //! return rcp ptr to neumann loads vector
    std::shared_ptr<Core::LinAlg::Vector<double>> get_neumann_loads_ptr() override
    {
      return neumann_loads_;
    }

    //! return true if an external force is applied to the system
    [[nodiscard]] bool has_external_force() const { return has_external_force_; }

    //! returns if restart information is needed for the current time step
    [[nodiscard]] bool is_restart_step() const
    {
      // write restart info if the simulation ends
      const bool is_finished = not not_finished();

      return (step_ % uprestart_ == 0 and step_ != 0) or (is_finished and step_ != 0);
    }

    //! returns if output of results is needed for the current time step
    [[nodiscard]] bool is_result_step() const { return (step_ % upres_ == 0) or is_restart_step(); }

    /*========================================================================*/
    //! @name turbulence and related
    /*========================================================================*/

    ///! get access to dynamic Smagorinsky class of fluid time integration
    void access_dyn_smag_filter(std::shared_ptr<FLD::DynSmagFilter> dynSmag);
    ///! get access to dynamic Vreman class of fluid time integration
    void access_vreman(std::shared_ptr<FLD::Vreman> vrem);

    ///! calculate intermediate solution to determine forcing for homogeneous isotropic turbulence
    void calc_intermediate_solution();

    /*========================================================================*/
    //! @name  fs3i methods
    /*========================================================================*/

    //! compute contribution of permeable surface/interface
    void surface_permeability(
        std::shared_ptr<Core::LinAlg::SparseOperator> matrix,  //!< system matrix
        std::shared_ptr<Core::LinAlg::Vector<double>> rhs      //!< rhs vector
    );

    //! interface for fps3i problem
    void kedem_katchalsky(std::shared_ptr<Core::LinAlg::SparseOperator> matrix,  //!< system matrix
        std::shared_ptr<Core::LinAlg::Vector<double>> rhs                        //!< rhs vector
    );

    /*========================================================================*/
    //! @name Biofilm methods
    /*========================================================================*/

    //! return scatra structure growth vector
    [[nodiscard]] std::shared_ptr<const Core::LinAlg::MultiVector<double>> str_growth() const
    {
      return scstrgrdisp_;
    }

    //! return scatra fluid growth vector
    [[nodiscard]] std::shared_ptr<const Core::LinAlg::MultiVector<double>> fld_growth() const
    {
      return scfldgrdisp_;
    }

    //! set scatra fluid displacement vector due to biofilm growth
    void set_sc_fld_gr_disp(
        std::shared_ptr<Core::LinAlg::MultiVector<double>> scatra_fluid_growth_disp);

    //! set scatra structure displacement vector due to biofilm growth
    void set_sc_str_gr_disp(
        std::shared_ptr<Core::LinAlg::MultiVector<double>> scatra_struct_growth_disp);

    //! set ptr to wrapper of this time integrator
    void set_model_evaluatro_ptr(Adapter::AdapterScatraWrapper* adapter_scatra_wrapper)
    {
      additional_model_evaluator_ = adapter_scatra_wrapper;
    }

    //! set the visualization writer
    void set_visualization_writer(
        std::shared_ptr<Core::IO::DiscretizationVisualizationWriterMesh> visualization_writer)
    {
      visualization_writer_ = std::move(visualization_writer);
    }

    //! return the visualization writer
    [[nodiscard]] Core::IO::DiscretizationVisualizationWriterMesh& visualization_writer() const
    {
      return *visualization_writer_;
    }

   protected:
    //! create vectors for Krylov projection if necessary
    void prepare_krylov_projection();

    /*========================================================================*/
    //! @name set element parameters
    /*========================================================================*/

    virtual void set_element_time_parameter(bool forcedincrementalsolver = false) const = 0;

    //! Set backward Euler time parameter
    virtual void set_element_time_parameter_backward_euler() const {};

    //! set time for evaluation of Neumann boundary conditions
    virtual void set_time_for_neumann_evaluation(Teuchos::ParameterList& params) = 0;

    //! Set general element parameters
    void set_element_general_parameters(bool calcinitialtimederivative = false) const;

    //! Set node set parameters
    void set_element_nodeset_parameters() const;

    //! Set additional problem-specific parameters for non-standard scalar transport problems
    //! (electrochemistry etc.)
    virtual void set_element_specific_scatra_parameters(Teuchos::ParameterList& eleparams) const {};

    //! Set element parameter specific for turbulence
    void set_element_turbulence_parameters(bool calcinitialtimederivative = false) const;

    /*========================================================================*/
    //! @name general framework
    /*========================================================================*/

    /*--- set, prepare, and predict ------------------------------------------*/

    //! compute history vector, i.e., the history part of the right-hand side vector with all
    //! contributions from the previous time step
    virtual void set_old_part_of_righthandside();

    //! create Krylov space projector
    void setup_krylov_space_projection(const Core::Conditions::Condition* kspcond);
    //! update Krylov space projector
    void update_krylov_space_projection();

    //! compute approximation for fluxes and add it to a parameter list
    void add_flux_approx_to_parameter_list(Teuchos::ParameterList& p) const;

    //! calculate consistent initial scalar time derivatives in compliance with initial scalar field
    //! this function is never called directly, but only via overloading
    virtual void calc_initial_time_derivative();

    //! initialize meshtying strategy (including standard case without meshtying)
    virtual void create_meshtying_strategy();

    /*--- calculate and update -----------------------------------------------*/

    //! apply Dirichlet boundary conditions to linear system of equations
    void apply_dirichlet_to_system();

    //! Apply Dirichlet boundary conditions on provided state vector
    virtual void apply_dirichlet_bc(const double time,  //!< evaluation time
        std::shared_ptr<Core::LinAlg::Vector<double>>
            phinp,  //!< transported scalar(s) (may be = null)
        std::shared_ptr<Core::LinAlg::Vector<double>>
            phidt  //!< first time derivative (may be = null)
    );

    //! compute outward pointing unit normal vectors for given conditions
    [[nodiscard]] std::shared_ptr<Core::LinAlg::MultiVector<double>> compute_normal_vectors(
        const std::vector<std::string>& condnames) const;

    //! evaluate Neumann inflow boundary condition
    void compute_neumann_inflow(std::shared_ptr<Core::LinAlg::SparseOperator> matrix,  //!< ?
        std::shared_ptr<Core::LinAlg::Vector<double>> rhs                              //!< ?
    );

    //! evaluate boundary condition due to convective heat transfer
    void evaluate_convective_heat_transfer(
        std::shared_ptr<Core::LinAlg::SparseOperator> matrix,  //!< ?
        std::shared_ptr<Core::LinAlg::Vector<double>> rhs      //!< ?
    );

    //! potential residual scaling and potential addition of Neumann terms
    void scaling_and_neumann();

    //! add actual Neumann loads multiplied with time factor to the residual
    virtual void add_neumann_to_residual() = 0;

    //! evaluate Neumann boundary conditions
    virtual void apply_neumann_bc(
        const std::shared_ptr<Core::LinAlg::Vector<double>>& neumann_loads  //!< Neumann loads
    );

    //! add parameters depending on the problem, i.e., loma, level-set, ...
    virtual void add_problem_specific_parameters_and_vectors(
        Teuchos::ParameterList& params  //!< parameter list
    );

    //! return the right time-scaling-factor for the true residual
    [[nodiscard]] virtual double residual_scaling() const = 0;

    //! solve linear system
    void linear_solve();

    //! contains the nonlinear iteration loop
    virtual void nonlinear_solve();

    //! contains the nonlinear iteration loop for truly partitioned multi-scale simulations
    void nonlinear_multi_scale_solve();

    //! solve micro scale in truly partitioned multi-scale simulations
    void nonlinear_micro_scale_solve();


    //! Calculate the reconstructed nodal gradient of phi by means of SPR
    std::shared_ptr<Core::LinAlg::MultiVector<double>> compute_superconvergent_patch_recovery(
        const Core::LinAlg::Vector<double>& state, const std::string& statename, int numvec,
        Teuchos::ParameterList& params, int dim) const;

    //! compute contributions of solution-depending boundary and interface conditions to global
    //! system of equations
    virtual void evaluate_solution_depending_conditions(
        std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,  //!< system matrix
        std::shared_ptr<Core::LinAlg::Vector<double>> rhs            //!< rhs vector
    );

    //! compute contribution of Robin boundary condition to eq. system
    void evaluate_robin_boundary_conditions(
        std::shared_ptr<Core::LinAlg::SparseOperator> matrix,  //!< system matrix
        std::shared_ptr<Core::LinAlg::Vector<double>> rhs      //!< rhs vector
    );

    //! compute contributions of additional solution-depending models to global system of equations
    virtual void evaluate_additional_solution_depending_models(
        std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,  //!< system matrix
        std::shared_ptr<Core::LinAlg::Vector<double>> rhs            //!< rhs vector
    );

    //! perform Aitken relaxation
    virtual void perform_aitken_relaxation(
        Core::LinAlg::Vector<double>& phinp,  //!< state vector to be relaxed
        const Core::LinAlg::Vector<double>&
            phinp_inc_diff  //!< difference between current and previous state vector increments
    );

    /*--- query and output ---------------------------------------------------*/

    //! returns true if \ref setup() was called and is still valid
    [[nodiscard]] bool is_setup() const { return issetup_; }

    //! returns true if \ ref init() was called and is still valid
    [[nodiscard]] bool is_init() const { return isinit_; }

    //! check if \ref setup() was called
    void check_is_setup() const;

    //! check if \ref init() was called
    void check_is_init() const;

    //! helper function to get algorithm title
    std::string map_tim_int_enum_to_string(Inpar::ScaTra::TimeIntegrationScheme term  //!< the enum
    );

    //! do we need a statistical sampling for boundary flux at the current time step?
    [[nodiscard]] bool do_boundary_flux_statistics() const
    {
      return ((step_ >= samstart_) and (step_ <= samstop_) and
              ((calcflux_boundary_ == Inpar::ScaTra::flux_total) or
                  (calcflux_boundary_ == Inpar::ScaTra::flux_diffusive) or
                  (calcflux_boundary_ == Inpar::ScaTra::flux_convective)));
    }

    //! write state vectors (phinp and convective velocity) to Gmsh postprocessing files
    void output_to_gmsh(int step, double time) const;

    /*!
     * @brief collect flux vectors for runtime output
     *
     * @param[in] flux      flux vector
     * @param[in] fluxtype  flux type ("domain" or "boundary")
     */
    virtual void collect_output_flux_data(
        std::shared_ptr<Core::LinAlg::MultiVector<double>> flux, const std::string& fluxtype);

    /*========================================================================*/
    //! @name Time, time-step and related methods
    /*========================================================================*/

    /*--- set, prepare, and predict ------------------------------------------*/

    //! adapt time step size if desired
    void adapt_time_step_size();

    //! compute time step size
    virtual void compute_time_step_size(double& dt);

    //! increment time and step value
    void increment_time_and_step();

    /*--- calculate and update -----------------------------------------------*/


    /*--- query and output ---------------------------------------------------*/

    /*========================================================================*/
    //! @name scalar degrees of freedom and related
    /*========================================================================*/

    /*--- set, prepare, and predict ------------------------------------------*/
    //! compute null space information associated with global system matrix if applicable
    void compute_null_space_if_necessary() const;

    //! create scalar handler
    virtual void create_scalar_handler();

    /*--- calculate and update -----------------------------------------------*/

    /*--- query and output ---------------------------------------------------*/

    /*========================================================================*/
    //! @name AVM3 and related
    /*========================================================================*/

    /*--- set, prepare, and predict ------------------------------------------*/

    //! prepare AVM3-based scale separation
    void avm3_preparation();

    //! AVM3-based scale separation
    virtual void avm3_separation() = 0;

    /*--- calculate and update -----------------------------------------------*/

    //! scaling of AVM3-based subgrid-diffusivity matrix
    void avm3_scaling(Teuchos::ParameterList& eleparams  //!< parameter list
    );

    /*--- query and output ---------------------------------------------------*/

    /*========================================================================*/
    //! @name turbulence and related
    /*========================================================================*/

    //! dynamic Smagorinsk model
    virtual void dynamic_computation_of_cs() = 0;

    //! dynamic Vreman model
    virtual void dynamic_computation_of_cv() = 0;

    //! calculate mean CsgsB to estimate CsgsD for multifractal subgrid-scale model
    void recompute_mean_csgs_b();

    /*========================================================================*/
    //! @name  not classified method - to be kept clean!!!
    /*========================================================================*/

    /*!
     * \brief Extract the Dirichlet toggle vector based on Dirichlet BC maps
     *
     * This method provides backward compatibility only. Formerly, the Dirichlet
     * conditions were handled with the Dirichlet toggle vector. Now, they are
     * stored and applied with maps, ie #dbcmaps_. Eventually, this method will
     * be removed.
     * note: VM3 solver still needs an explicit toggle vector for construction
     */
    [[nodiscard]] std::shared_ptr<const Core::LinAlg::Vector<double>> dirichlet_toggle() const;

    /*========================================================================*/

    /*========================================================================*/
    //! @name general framework variables
    /*========================================================================*/

    //! problem
    Global::Problem* const problem_;

    //! problem number
    const int probnum_;

    //! solver
    std::shared_ptr<Core::LinAlg::Solver> solver_;

    //! parameter list
    const std::shared_ptr<Teuchos::ParameterList> params_;

    //! parameter list containing extra parameters (application dependent)
    const std::shared_ptr<Teuchos::ParameterList> extraparams_;

    //! processor id
    int myrank_;

    //! Extractor used for convergence check either in ELCH or LOMA case
    std::shared_ptr<Core::LinAlg::MapExtractor> splitter_;

    //! meshtying strategy (includes standard case without meshtying)
    std::shared_ptr<ScaTra::MeshtyingStrategyBase> strategy_;

    //! Ptr to time integration wrapper.
    //! That wrapper holds a ptr to this time integrator in turn.
    //! This Ptr is uneqal nullptr only if a scatra adapter was constructed.
    //! Class AdapterScatraWrapper sets this pointer during construction
    //! by calling \ref set_model_evaluatro_ptr .
    Adapter::AdapterScatraWrapper* additional_model_evaluator_;

    /*========================================================================*/
    //! @name flags and enums
    /*========================================================================*/

    //! flag for Eulerian or ALE formulation of equation(s)
    bool isale_;

    //! solvertype and flags for nonlinear (always incremental) and (linear) incremental solver
    Inpar::ScaTra::SolverType solvtype_;

    //! type of equilibration of global system of scalar transport equations
    const Core::LinAlg::EquilibrationMethod equilibrationmethod_;

    //! type of global system matrix in global system of equations
    const Core::LinAlg::MatrixType matrixtype_;

    //! incremental or linear full solving? rename -> is_incremental_
    bool incremental_;

    //! flag for fine-scale subgrid-viscosity
    Inpar::ScaTra::FSSUGRDIFF fssgd_;

    //! LOMA-specific parameter: turbulence model
    Inpar::FLUID::TurbModelAction turbmodel_;

    //! flag indicating availability of scatra-scatra interface kinetics condition(s)
    bool s2ikinetics_;

    //! flag for scatra-scatra interface mesh tying
    bool s2imeshtying_;

    //! flag for artery-scatra interface coupling
    const bool arterycoupling_;

    //! flag for scatra-scatra heterogeneous reaction coupling
    const bool heteroreaccoupling_;

    //! flag for macro scale in multi-scale simulations
    const bool macro_scale_;

    //! flag for micro scale in multi-scale simulations
    const bool micro_scale_;

    //! flag for external force
    bool has_external_force_;

    /*--- query and output ---------------------------------------------------*/

    //! flag for calculating flux vector field inside domain
    Inpar::ScaTra::FluxType calcflux_domain_;

    //! flag for approximate domain flux calculation involving matrix lumping
    const bool calcflux_domain_lumped_;

    //! flag for calculating flux vector field on boundary
    Inpar::ScaTra::FluxType calcflux_boundary_;

    //! flag for approximate boundary flux calculation involving matrix lumping
    const bool calcflux_boundary_lumped_;

    //! ids of scalars for which flux vectors are written (starting with 1)
    std::shared_ptr<std::vector<int>> writefluxids_;

    //! flux vector field inside domain
    std::shared_ptr<Core::LinAlg::MultiVector<double>> flux_domain_;

    //! flux vector field on boundary
    std::shared_ptr<Core::LinAlg::MultiVector<double>> flux_boundary_;

    //! map extractor associated with boundary segments for flux calculation
    std::shared_ptr<Core::LinAlg::MultiMapExtractor> flux_boundary_maps_;

    //! vector for statistical evaluation of normal fluxes
    std::shared_ptr<Core::LinAlg::SerialDenseVector> sumnormfluxintegral_;

    //! the last step number when fluxes have been computed
    int lastfluxoutputstep_;

    //! boolean to write the material id of each element (input parameter)
    const bool output_element_material_id_;
    //! flag for printing out total and mean values of transported scalars
    const Inpar::ScaTra::OutputScalarType outputscalars_;

    //! boolean to write Gmsh postprocessing files (input parameter)
    const bool outputgmsh_;

    //! boolean to write state vector to matlab file (input parameter)
    const bool output_state_matlab_;

    //! flag for finite difference check
    const Inpar::ScaTra::FdCheck fdcheck_;

    //! perturbation magnitude for finite difference check
    const double fdcheckeps_;

    //! relative tolerance for finite difference check
    const double fdchecktol_;

    //! flag for computation of domain and boundary integrals, i.e., of surface areas and volumes
    //! associated with specified nodesets
    const Inpar::ScaTra::ComputeIntegrals computeintegrals_;

    //! flag for calculation of relative error with reference to analytical solution
    const Inpar::ScaTra::CalcError calcerror_;

    /*========================================================================*/
    //! @name Time, time-step, and iteration variables
    /*========================================================================*/

    //! actual time
    double time_;

    //! maximum simulation time
    double maxtime_;

    //! actual step number
    int step_;

    //! maximum number of steps ? name maxtime vs. stepmax
    int stepmax_;

    //! time step size
    double dta_;

    //! time measurement element
    double dtele_;

    //! time measurement solve
    double dtsolve_;

    //! number of newton iterations in actual timestep
    int iternum_;

    //! number of outer iterations in partitioned simulations
    unsigned iternum_outer_;

    //! used time integration scheme
    Inpar::ScaTra::TimeIntegrationScheme timealgo_;

    /*========================================================================*/
    //! @name scalar degrees of freedom variables
    /*========================================================================*/

    //! number of space dimensions
    int nsd_;

    //! scalar manager
    std::shared_ptr<ScalarHandler> scalarhandler_;

    //! scalar manager
    std::shared_ptr<OutputScalarsStrategyBase> outputscalarstrategy_;

    //! domain integral manager
    std::shared_ptr<OutputDomainIntegralStrategy> outputdomainintegralstrategy_;

    //! stores the components phi is composed of necessary for the output
    std::vector<std::optional<std::string>> phi_components_{};

    //! phi at time n
    std::shared_ptr<Core::LinAlg::Vector<double>> phin_;
    //! phi at time n+1
    std::shared_ptr<Core::LinAlg::Vector<double>> phinp_;
    //! increment of phi at time n+1 for partitioned simulations
    std::shared_ptr<Core::LinAlg::Vector<double>> phinp_inc_;
    //! increment of phi at time n+1 from previous outer iteration step for partitioned simulations
    std::shared_ptr<Core::LinAlg::Vector<double>> phinp_inc_old_;
    //! relaxation parameters
    std::vector<double> omega_;

    //! time derivative of phi at time n
    std::shared_ptr<Core::LinAlg::Vector<double>> phidtn_;
    //! time derivative of phi at time n+1
    std::shared_ptr<Core::LinAlg::Vector<double>> phidtnp_;

    //! histvector --- a linear combination of phinm, phin (BDF)
    //!                or phin, phidtn (One-Step-Theta)
    std::shared_ptr<Core::LinAlg::Vector<double>> hist_;

    //! density at time n+alpha_F (gen-alpha) or n+1 (otherwise) for natural convection algorithm
    std::shared_ptr<Core::LinAlg::Vector<double>> densafnp_;

    //! relative errors of scalar fields in L2 and H1 norms
    std::shared_ptr<std::vector<double>> relerrors_;

    /*========================================================================*/
    //! @name velocity, pressure, and related
    /*========================================================================*/

    //! subgrid-scale velocity required for multifractal subgrid-scale modeling
    std::shared_ptr<Core::LinAlg::MultiVector<double>> fsvel_;

    //! type of velocity field
    const Inpar::ScaTra::VelocityField velocity_field_type_;

    //! mean in time at the interface concentration
    std::shared_ptr<const Core::LinAlg::Vector<double>> mean_conc_;

    //! Membrane concentration in interface between a scatracoupling (needed for instance for type
    //! fps3i)
    std::shared_ptr<const Core::LinAlg::Vector<double>> membrane_conc_;

    //! mean concentration of micro discretization  on macro dis
    std::shared_ptr<Core::LinAlg::Vector<double>> phinp_micro_;

   private:
    //! number of dofset associated with displacement dofs
    int nds_disp_;

    //! number of dofset associated with interface growth dofs
    int nds_growth_;

    //! number of dofset to write micro scale values on
    int nds_micro_;

    //! number of dofset associated with pressure dofs
    int nds_pres_;

    //! number of dofset associated with scatra dofs
    int nds_scatra_;

    //! number of dofset associated with thermo dofs
    int nds_thermo_;

    //! number of dofset associated with two-tensor quantity dofs, e.g. stresses, strains
    int nds_two_tensor_quantity_;

    //! number of dofset associated with velocity related dofs
    int nds_vel_;

    //! number of dofset associated with wall shear stress dofs
    int nds_wss_;

    /*========================================================================*/
    //! @name coefficients and related
    /*========================================================================*/
   protected:
    //! subgrid-diffusivity(-scaling) vector
    std::shared_ptr<Core::LinAlg::Vector<double>> subgrdiff_;

    //! densification coefficients for natural convection
    std::vector<double> densific_;

    //! initial concentrations for natural convection
    std::vector<double> c0_;

    //! reaction coefficient
    double macro_micro_rea_coeff_;

    /*========================================================================*/
    //! @name Galerkin discretization, boundary conditions, and related
    /*========================================================================*/

    //! the scalar transport discretization
    std::shared_ptr<Core::FE::Discretization> discret_;

    //! the discretization writer
    std::shared_ptr<Core::IO::DiscretizationWriter> output_;

    //! form of convective term
    Inpar::ScaTra::ConvForm convform_;

    //! system matrix (either sparse matrix or block sparse matrix)
    std::shared_ptr<Core::LinAlg::SparseOperator> sysmat_;

    //! map extractor associated with the degrees of freedom inside the blocks of global system
    //! matrix
    std::shared_ptr<Core::LinAlg::MultiMapExtractor> dof_block_maps_;

    //! map extractor associated with the nodes inside the blocks of global system matrix
    std::shared_ptr<Core::LinAlg::MultiMapExtractor> node_block_maps_;

    //! a vector of zeros to be used to enforce zero dirichlet boundary conditions
    std::shared_ptr<Core::LinAlg::Vector<double>> zeros_;

    //! function to set external force
    std::function<void()> set_external_force_;

    //! maps for extracting Dirichlet and free DOF sets
    std::shared_ptr<Core::LinAlg::MapExtractor> dbcmaps_;

    //! the vector containing body and surface forces
    std::shared_ptr<Core::LinAlg::Vector<double>> neumann_loads_;

    //! unit outer normal vector field for flux output
    std::shared_ptr<Core::LinAlg::MultiVector<double>> normals_;

    //! residual vector
    std::shared_ptr<Core::LinAlg::Vector<double>> residual_;

    //! true (rescaled) residual vector without zeros at Dirichlet conditions
    std::shared_ptr<Core::LinAlg::Vector<double>> trueresidual_;

    //! nonlinear iteration increment vector
    std::shared_ptr<Core::LinAlg::Vector<double>> increment_;

    //! options for meshtying
    Inpar::FLUID::MeshTying msht_;

    /*========================================================================*/
    //! @name AVM3 variables
    /*========================================================================*/

    //! only necessary for AVM3: fine-scale subgrid-diffusivity matrix
    std::shared_ptr<Core::LinAlg::SparseMatrix> sysmat_sd_;

    //! only necessary for AVM3: scale-separation matrix ? rename small caps
    std::shared_ptr<Core::LinAlg::SparseMatrix> Sep_;

    //! only necessary for AVM3: normalized fine-scale subgrid-viscosity matrix ? rename small caps
    std::shared_ptr<Core::LinAlg::SparseMatrix> Mnsv_;

    /*========================================================================*/
    //! @name turbulent flow variables
    /*========================================================================*/

    //! one instance of the filter object
    std::shared_ptr<FLD::DynSmagFilter> DynSmag_;
    std::shared_ptr<FLD::Vreman> Vrem_;

    //! parameters for sampling/dumping period
    int samstart_;
    int samstop_;
    int dumperiod_;

    //! flag for turbulent inflow (turbulent loma specific)
    bool turbinflow_;

    //! number of inflow generation time steps
    int numinflowsteps_;

    /// flag for special turbulent flow
    std::string special_flow_;

    //! the vector containing source term externally computed
    //! forcing for homogeneous isotropic turbulence
    std::shared_ptr<Core::LinAlg::Vector<double>> forcing_;

    //! forcing for homogeneous isotropic turbulence
    std::shared_ptr<ScaTra::HomoIsoTurbScalarForcing> homisoturb_forcing_;

    /*========================================================================*/
    //! @name variables for orthogonal space projection aka Krylov projection
    /*========================================================================*/

    bool updateprojection_;  //!< bool triggering update of Krylov projection
    std::shared_ptr<Core::LinAlg::KrylovProjector> projector_;  //!< Krylov projector himself

    /*========================================================================*/
    //! @name not classified variables - to be kept clean!!!
    /*========================================================================*/

    //! write results every upres_ steps ? writesolutionevery_
    int upres_;

    //! write restart data every uprestart_ steps ? writesolutioneveryrestart_
    int uprestart_;

    //! flag for potential Neumann inflow boundary condition
    bool neumanninflow_;

    //! flag for potential boundary condition due to convective heat transfer
    bool convheatrans_;

    //! macro-scale state variables for macro-micro coupling in multi-scale simulations
    std::vector<double> phinp_macro_;

    //! micro-scale coupling flux for macro-micro coupling in multi-scale simulations
    double q_;

    //! derivatives of micro-scale coupling flux w.r.t. macro-scale state variables for macro-micro
    //! coupling in multi-scale simulations
    std::vector<double> dq_dphi_;

    /*========================================================================*/

    /*========================================================================*/
    //! @name Biofilm specific stuff
    /*========================================================================*/

    // TODO: SCATRA_ELE_CLEANING: BIOFILM
    //! scatra fluid displacement due to growth
    std::shared_ptr<Core::LinAlg::MultiVector<double>> scfldgrdisp_;

    //! scatra structure displacement due to growth
    std::shared_ptr<Core::LinAlg::MultiVector<double>> scstrgrdisp_;

    //! flag for printing out integral values of reaction
    const bool outintegrreac_;

    //! @name Nitsche contact stuff
    //@{

    //! nitsche contact strategy
    std::shared_ptr<CONTACT::NitscheStrategySsi> contact_strategy_nitsche_;

    //@}

   private:
    /*========================================================================*/
    //! @name flags and enums
    /*========================================================================*/

    //! flag for potentially skipping computation of initial time derivative
    bool skipinitder_;

    //! flag indicating if time step was changed
    bool timestepadapted_;

    //! pointer to visualization writer object
    std::shared_ptr<Core::IO::DiscretizationVisualizationWriterMesh> visualization_writer_;

    //! flag indicating if class is setup
    bool issetup_;

    //! flag indicating if class is initialized
    bool isinit_;

    /*========================================================================*/
    //! @name general framework
    /*========================================================================*/

    //! set flag true after setup or false if setup became invalid
    void set_is_setup(bool trueorfalse) { issetup_ = trueorfalse; };

    //! set flag true after init or false if init became invalid
    void set_is_init(bool trueorfalse) { isinit_ = trueorfalse; };

  };  // class ScaTraTimIntImpl

  /*========================================================================*/
  /*========================================================================*/
  /*!
   * \brief Helper class for managing different number of degrees of freedom per node
   */
  class ScalarHandler
  {
   public:
    /*========================================================================*/
    //! @name Constructors and destructors and related methods
    /*========================================================================*/

    //! Standard Constructor
    ScalarHandler() = default;

    /**
     * Virtual destructor.
     */
    virtual ~ScalarHandler() = default;

    //! set up scalar handler
    virtual void setup(const ScaTraTimIntImpl* const scatratimint);

    /*========================================================================*/
    //! @name Access and Query methods
    /*========================================================================*/

    //! return maximum number of dofs per node
    [[nodiscard]] int num_dof_per_node_in_condition(const Core::Conditions::Condition& condition,
        const Core::FE::Discretization& discret) const;

    //! return maximum number of transported scalars per node
    [[nodiscard]] virtual int num_scal_in_condition(const Core::Conditions::Condition& condition,
        const std::shared_ptr<const Core::FE::Discretization>& discret) const
    {
      return num_dof_per_node_in_condition(condition, *discret);
    }

    //! return number of dofs per node
    [[nodiscard]] virtual int num_dof_per_node() const;

    //! return maximum number of dofs per node
    [[nodiscard]] int max_num_dof_per_node() const;

    //! return maximum number of transported scalars per node
    [[nodiscard]] virtual int num_scal() const { return num_dof_per_node(); }

    //! return flag indicating equal number of DOFs per node in whole discretization
    [[nodiscard]] bool equal_num_dof() const { return equalnumdof_; }

   protected:
    /*========================================================================*/
    //! @name general framework
    /*========================================================================*/
    //! check if \ref setup() was called
    void check_is_setup() const;

    /*========================================================================*/
    //! @name Internal variables
    /*========================================================================*/
    //! number of transported scalars
    std::set<int> numdofpernode_;

    //! flag indicating equal number of DOFs per node in whole discretization
    bool equalnumdof_;

   private:
    /*========================================================================*/
    //! @name Internal variables
    /*========================================================================*/
    //! flag indicating \ref setup() call
    bool issetup_;
  };

  /*========================================================================*/
  /*========================================================================*/
  /*!
   * \brief Base class for output of mean and total scalar values
   */
  class OutputScalarsStrategyBase
  {
   public:
    //! constructor
    OutputScalarsStrategyBase(const ScaTraTimIntImpl* scatratimint);

    /**
     * Virtual destructor.
     */
    virtual ~OutputScalarsStrategyBase() = default;

    //! do the output
    void output_total_and_mean_scalars(const ScaTraTimIntImpl* scatratimint, int num);

    /*========================================================================*/
    //! @name Access methods
    /*========================================================================*/

    //! return total values of transported scalars
    [[nodiscard]] const std::map<const int, std::vector<double>>& total_scalars() const
    {
      return totalscalars_;
    }

    //! return mean values of transported scalars
    [[nodiscard]] const std::map<const int, std::vector<double>>& mean_scalars() const
    {
      return meanscalars_;
    }

   protected:
    /*========================================================================*/
    //! @name Helper methods
    /*========================================================================*/

    //! evaluate mean and total scalars and print them to file and screen
    virtual void evaluate_integrals(const ScaTraTimIntImpl* scatratimint) = 0;

    //! print bar to screen as bottom of table
    void finalize_screen_output() const;

    //! evaluate csv data and return it in a map
    virtual std::map<std::string, std::vector<double>> prepare_csv_output() = 0;

    //! fill parameter list and set variables in discretization for evaluation of mean scalars
    void prepare_evaluate(
        const ScaTraTimIntImpl* scatratimint, Teuchos::ParameterList& eleparams) const;

    //! print header of table for summary of mean values to screen
    void print_header_to_screen(const std::string& dis_name) const;

    //! Print evaluated data to screen
    virtual void print_to_screen() = 0;

    /*========================================================================*/
    //! @name Internal variables
    /*========================================================================*/
    //! size of domain
    std::map<const int, double> domainintegral_;

    //! mean values of transported scalars
    std::map<const int, std::vector<double>> meanscalars_;

    //! mean values of gradient of transported scalars
    std::map<const int, std::vector<double>> meangradients_;

    //! mean of micro scalars
    std::map<const int, std::vector<double>> micromeanscalars_;

    //! process number
    int myrank_;

    //! do output of mean of gradient
    bool output_mean_grad_;

    //! do output of micro dis
    bool output_micro_dis_;

    //! writes evaluated data to output
    std::optional<Core::IO::RuntimeCsvWriter> runtime_csvwriter_;

    //! total values of transported scalars
    std::map<const int, std::vector<double>> totalscalars_;
  };

  /*!
   * \brief Strategy evaluating total and mean scalars on entire domain
   */
  class OutputScalarsStrategyDomain : virtual public OutputScalarsStrategyBase
  {
   public:
    OutputScalarsStrategyDomain(const ScaTraTimIntImpl* scatratimint);

   protected:
    void evaluate_integrals(const ScaTraTimIntImpl* scatratimint) override;

    std::map<std::string, std::vector<double>> prepare_csv_output() override;

    void print_to_screen() override;

   private:
    const int dummy_domain_id_;
    //! number of degrees of freedom per node
    int numdofpernode_;

    //! number of transported scalars
    int numscal_;
  };

  /*========================================================================*/
  /*========================================================================*/
  /*!
   * \brief Strategy evaluating total and mean scalars on given condition
   */
  class OutputScalarsStrategyCondition : virtual public OutputScalarsStrategyBase
  {
   public:
    OutputScalarsStrategyCondition(const ScaTraTimIntImpl* scatratimint);

   protected:
    void evaluate_integrals(const ScaTraTimIntImpl* scatratimint) override;

    std::map<std::string, std::vector<double>> prepare_csv_output() override;

    void print_to_screen() override;

   private:
    //! vector of 'TotalAndMeanScalar'-conditions
    std::vector<const Core::Conditions::Condition*> conditions_;

    //! number of degrees of freedom per node per 'TotalAndMeanScalar'-conditions
    std::map<int, int> numdofpernodepercondition_;

    //! number of scalars per 'TotalAndMeanScalar'-conditions
    std::map<int, int> numscalpercondition_;
  };

  /*========================================================================*/
  /*========================================================================*/
  /*!
   * \brief Strategy evaluating total and mean scalars on entire domain and on given condition
   */
  class OutputScalarsStrategyDomainAndCondition : public OutputScalarsStrategyDomain,
                                                  public OutputScalarsStrategyCondition
  {
   public:
    OutputScalarsStrategyDomainAndCondition(const ScaTraTimIntImpl* scatratimint);

   protected:
    void evaluate_integrals(const ScaTraTimIntImpl* scatratimint) override;

    std::map<std::string, std::vector<double>> prepare_csv_output() override;

    void print_to_screen() override;
  };

  /*========================================================================*/
  /*========================================================================*/
  /*!
   * \brief Strategy evaluating domain integrals on given condition
   */
  class OutputDomainIntegralStrategy
  {
   public:
    //! Standard Constructor
    explicit OutputDomainIntegralStrategy(const ScaTraTimIntImpl* scatratimint);

    //! evaluate domain integrals and print to screen
    void evaluate_integrals_and_print_results(
        const ScaTraTimIntImpl* scatratimint, const std::string& condstring);

    /*========================================================================*/
    //! @name Access methods
    /*========================================================================*/

    //! return values of domain integrals
    [[nodiscard]] const std::vector<double>& domain_integrals() const
    {
      return domainintegralvalues_;
    }

    //! return values of boundary
    [[nodiscard]] const std::vector<double>& boundary_integrals() const
    {
      return boundaryintegralvalues_;
    }

   private:
    //! vector of 'DomainIntegral'-conditions
    std::vector<const Core::Conditions::Condition*> conditionsdomain_;
    //! vector of 'BoundaryIntegral'-conditions
    std::vector<const Core::Conditions::Condition*> conditionsboundary_;
    //! vector of 'DomainIntegral'-values
    std::vector<double> domainintegralvalues_;
    //! vector of 'BoundaryIntegral'-values
    std::vector<double> boundaryintegralvalues_;
  };

}  // namespace ScaTra
FOUR_C_NAMESPACE_CLOSE

#endif
