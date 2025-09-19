// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_POROFLUID_PRESSURE_BASED_ALGORITHM_HPP
#define FOUR_C_POROFLUID_PRESSURE_BASED_ALGORITHM_HPP



#include "4C_config.hpp"

#include "4C_adapter_art_net.hpp"
#include "4C_adapter_porofluid_pressure_based.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_io_discretization_visualization_writer_mesh.hpp"
#include "4C_io_runtime_csv_writer.hpp"
#include "4C_linalg_map.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_porofluid_pressure_based_input.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

FOUR_C_NAMESPACE_OPEN


/*==========================================================================*/
// forward declarations
/*==========================================================================*/

namespace Discret
{
  class ResultTest;
}  // namespace Discret

namespace Core::DOFSets
{
  class DofSet;
}  // namespace Core::DOFSets


namespace Core::IO
{
  class DiscretizationWriter;
}

namespace Core::LinAlg
{
  class Solver;
  class SparseMatrix;
  class MapExtractor;
  class BlockSparseMatrixBase;
  class SparseOperator;
  class KrylovProjector;
}  // namespace Core::LinAlg

namespace Adapter
{
  class ArtNet;
}


namespace PoroPressureBased
{
  // forward declaration
  class MeshtyingArtery;

  /*!
   * \brief implicit time integration for porous multiphase flow problems
   */

  class PorofluidAlgorithm : public Adapter::PoroFluidMultiphase
  {
   public:
    /*========================================================================*/
    //! @name Constructors and destructors and related methods
    /*========================================================================*/

    //! Standard Constructor
    PorofluidAlgorithm(std::shared_ptr<Core::FE::Discretization> dis, const int linsolvernumber,
        const Teuchos::ParameterList& probparams, const Teuchos::ParameterList& poroparams,
        std::shared_ptr<Core::IO::DiscretizationWriter> output);


    //! initialize time integration
    void init(bool isale, int nds_disp, int nds_vel, int nds_solidpressure, int nds_scalar,
        const std::map<int, std::set<int>>* nearby_ele_pairs) override;

    /*========================================================================*/
    //! @name general framework
    /*========================================================================*/

    /*--- set, prepare, and predict ------------------------------------------*/

    //! prepare time loop
    void prepare_time_loop() override;

    //! setup the variables to do a new time step
    void prepare_time_step() override;

    //! initialization procedure prior to evaluation of first time step
    virtual void prepare_first_time_step();

    //! initialization procedure prior to evaluation of first time step
    virtual void calc_initial_time_derivative();

    //! read restart data
    void read_restart(int step) override;

    /// create result test for porous fluid field
    std::shared_ptr<Core::Utils::ResultTest> create_field_test() override;

    //! finite difference check for system matrix
    void fd_check();

    /*--- calculate and update -----------------------------------------------*/

    //! do time integration (time loop)
    void time_loop() override;

    //! general solver call for coupled algorithms
    void solve() override;

    //! update the solution after convergence of the nonlinear iteration.
    void update() override;

    ///  compute time derivative
    void compute_time_derivative();

    /// add parameters specific for time-integration scheme
    void add_time_integration_specific_vectors();

    //! apply moving mesh data
    void apply_mesh_movement(
        std::shared_ptr<const Core::LinAlg::Vector<double>> dispnp  //!< displacement vector
        ) override;

    //! set convective velocity field (+ pressure and acceleration field as
    //! well as fine-scale velocity field, if required)
    void set_velocity_field(
        std::shared_ptr<const Core::LinAlg::Vector<double>> vel  //!< velocity vector
        ) override;

    //! set state on discretization
    void set_state(unsigned nds, const std::string& name,
        std::shared_ptr<const Core::LinAlg::Vector<double>> state) override;

    //! calculate error compared to analytical solution
    void evaluate_error_compared_to_analytical_sol() override;

    /*--- query and output ---------------------------------------------------*/

    //! print information about current time step to screen
    virtual void print_time_step_info();

    //! iterative update of phinp
    void update_iter(
        const std::shared_ptr<const Core::LinAlg::Vector<double>> inc  //!< increment vector for phi
        ) override;

    //! build linear system tangent matrix, rhs/force residual
    void evaluate() override;

    //! apply Dirichlet Boundary Condition
    void prepare_system_for_newton_solve();

    //! direct access to system matrix
    std::shared_ptr<Core::LinAlg::SparseMatrix> system_matrix() override
    {
      return std::dynamic_pointer_cast<Core::LinAlg::SparseMatrix>(sysmat_);
    };

    //! Return MapExtractor for Dirichlet boundary conditions
    std::shared_ptr<const Core::LinAlg::MapExtractor> get_dbc_map_extractor() const override
    {
      return dbcmaps_with_volfracpress_;
    }

    //! right-hand side alias the dynamic force residual
    std::shared_ptr<const Core::LinAlg::Vector<double>> rhs() const override { return residual_; }

    //! right-hand side alias the dynamic force residual for coupled system
    std::shared_ptr<const Core::LinAlg::Vector<double>> artery_porofluid_rhs() const override;

    //! return discretization
    std::shared_ptr<Core::FE::Discretization> discretization() const override { return discret_; }

    //! access dof row map
    std::shared_ptr<const Core::LinAlg::Map> dof_row_map(unsigned nds) const override;

    //! access dof row map
    std::shared_ptr<const Core::LinAlg::Map> artery_dof_row_map() const override;

    //! direct access to block system matrix of artery poro problem
    std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> artery_porofluid_sysmat() const override;

    //! output solution and restart data to file
    void output() override;

    //! output solution and restart data to file
    virtual void print_header();

    //! write data required for restart
    virtual void output_restart();

    /*========================================================================*/
    //! @name Time, time-step and related methods
    /*========================================================================*/

    /*--- query and output ---------------------------------------------------*/

    //! return current time value
    double time() const { return time_; }

    //! return current step number
    int step() const { return step_; }

    //! return number of newton iterations in last timestep
    double iter_num() const { return iternum_; }

    //! return time step size
    double dt() const { return dt_; }

    /*========================================================================*/
    //! @name degrees of freedom and related
    /*========================================================================*/

    /*--- set, prepare, and predict ------------------------------------------*/

    //! set the initial field
    virtual void set_initial_field(InitialField init,  //!< type of initial field
        int startfuncno                                //!< number of spatial function
    );

    /*--- query and output ---------------------------------------------------*/

    //! return primary variable at time n+1
    std::shared_ptr<const Core::LinAlg::Vector<double>> phinp() const override { return phinp_; }

    //! return primary variable at time n
    std::shared_ptr<const Core::LinAlg::Vector<double>> phin() const override { return phin_; }

    //! return time derivative of the primary variable at time n
    std::shared_ptr<const Core::LinAlg::Vector<double>> phidtn() const { return phidtn_; }

    //! return time derivative of the primary variable at time n+1
    std::shared_ptr<const Core::LinAlg::Vector<double>> phidtnp() const { return phidtnp_; }

    //! return primary variable history
    std::shared_ptr<const Core::LinAlg::Vector<double>> hist() const { return hist_; }

    //! return solid pressure field
    std::shared_ptr<const Core::LinAlg::Vector<double>> solid_pressure() const override
    {
      if (!output_solidpress_)
        FOUR_C_THROW("solid pressure requested but flag OUTPUT_SOLIDPRESS set to no");
      return solidpressure_;
    }

    //! return pressure field
    std::shared_ptr<const Core::LinAlg::Vector<double>> pressure() const override
    {
      if (!output_satpress_)
        FOUR_C_THROW("pressure requested but flag OUTPUT_SATANDPRESS set to no");
      return pressure_;
    }

    //! return saturation field
    std::shared_ptr<const Core::LinAlg::Vector<double>> saturation() const override
    {
      if (!output_satpress_)
        FOUR_C_THROW("saturation requested but flag OUTPUT_SATANDPRESS set to no");
      return saturation_;
    }

    //! return phase flux field at time n+1
    std::shared_ptr<const Core::LinAlg::MultiVector<double>> flux() const override { return flux_; }

    //! return phase velocity at time n+1
    std::shared_ptr<const Core::LinAlg::MultiVector<double>> phase_velocity() const
    {
      return phase_velocities_;
    }

    //! return volfac_blood_lung at time n+1
    std::shared_ptr<Core::LinAlg::Vector<double>> volfrac_blood_lung() const
    {
      return volfrac_blood_lung_;
    }

    //! return determinant of deformation gradient at time n+1
    std::shared_ptr<Core::LinAlg::Vector<double>> det_def_grad() const { return det_def_grad_; }

    //! return number of dof set associated with solid pressure
    int get_dof_set_number_of_solid_pressure() const override { return nds_solidpressure_; };

    //! return valid volume fraction species
    std::shared_ptr<const Core::LinAlg::Vector<double>> valid_vol_frac_spec_dofs() const override
    {
      return valid_volfracspec_dofs_;
    }

    //! return number of domain integral functions
    int num_domain_int_functions() const { return num_domainint_funct_; }

    //! return the values of the domain integrals
    std::shared_ptr<const Core::LinAlg::SerialDenseVector> domain_int_values() const
    {
      return domain_integrals_;
    }

    //! return the meshtying strategy
    std::shared_ptr<PoroPressureBased::MeshtyingArtery> mesh_tying_strategy() const
    {
      return meshtying_;
    }

   protected:
    //! runtime CSV-writer for domain integrals
    std::optional<Core::IO::RuntimeCsvWriter> runtime_csvwriter_domain_integrals_;

   private:
    /// set time parameter for element evaluation (called before every time step)
    void set_element_time_step_parameter() const;

    //! set time for evaluation of Neumann boundary conditions
    void set_time_for_neumann_evaluation(Teuchos::ParameterList& params);

    //! Set general element parameters
    void set_element_general_parameters() const;

    //! Set the part of the residual vector belonging to the last timestep.
    void set_old_part_of_righthandside();

    /// do explicit predictor step (-> better starting value for nonlinear solver)
    void explicit_predictor();

    /*--- calculate and update -----------------------------------------------*/

    //! Apply Dirichlet boundary conditions on provided state vector
    void apply_dirichlet_bc(const double time,                //!< evaluation time
        std::shared_ptr<Core::LinAlg::Vector<double>> prenp,  //!< pressure (may be = null)
        std::shared_ptr<Core::LinAlg::Vector<double>>
            predt  //!< first time derivative (may be = null)
    );

    //! potential residual scaling and potential addition of Neumann terms
    void scaling_and_neumann();

    //! add actual Neumann loads multiplied with time factor to the residual
    virtual void add_neumann_to_residual();

    //! Apply Neumann boundary conditions
    void apply_neumann_bc(Core::LinAlg::Vector<double>& neumann_loads  //!< Neumann loads
    );

    //! call elements to calculate system matrix and rhs and assemble
    virtual void assemble_mat_and_rhs();

    //! call elements to find the valid volume frac pressures and species
    virtual void evaluate_valid_volume_frac_press_and_spec();

    //! apply the additional volume fraction pressures as DBC
    virtual void apply_additional_dbc_for_vol_frac_press();

    //! apply the starting Dirichlet boundary condition
    virtual void apply_starting_dbc();

    //! call elements to calculate fluid coupling matrix with structure and assemble
    void assemble_fluid_struct_coupling_mat(
        std::shared_ptr<Core::LinAlg::SparseOperator> k_fs) override;

    //! call elements to calculate fluid coupling matrix with scatra and assemble
    void assemble_fluid_scatra_coupling_mat(
        std::shared_ptr<Core::LinAlg::SparseOperator> k_pfs) override;

    //! return the right time-scaling-factor for the true residual
    double residual_scaling() const { return 1.0 / (dt_ * theta_); }

    //! contains the nonlinear iteration loop
    virtual void nonlinear_solve();

    //! check convergence (or divergence) of nonlinear iteration
    bool abort_nonlin_iter(const int itnum,  //!< current value of iteration step counter
        const int itemax,                    //!< maximum number of iteration steps
        const double abstolres,              //!< absolute tolerance for the residual norm
        double& actresidual                  //!< return value of the current residual
    );

    //! linear solve
    virtual void linear_solve(bool isadapttol, double actresidual, double adaptolbetter);

    //! reconstruct pressures and saturation from current solution
    void reconstruct_pressures_and_saturations() override;

    //! reconstruct solid pressures from current solution
    void reconstruct_solid_pressures();

    //! reconstruct fluxes from current solution
    void reconstruct_flux() override;

    //! calculate phase velocities from current solution
    void calculate_phase_velocities() override;

    //! reconstruct porosity from current solution
    void reconstruct_porosity();

    //! reconstruct volfrac blood lung from current solution
    void reconstruct_volfrac_blood_lung();

    //! reconstruct determinant of deformation gradient from current solution
    void reconstruct_determinant_of_derformation_gradient();

    //! evaluate domain integrals
    void evaluate_domain_integrals();

    /*--- query and output ---------------------------------------------------*/

    //! is output needed for the current time step?
    bool do_output() { return ((step_ % upres_ == 0) or (step_ % uprestart_ == 0)); };

    //! collect runtime output data
    void collect_runtime_output_data();

    //! print header of convergence table to screen
    virtual void print_convergence_header();

    //! print first line of convergence table to screen
    virtual void print_convergence_values_first_iter(
        const int& itnum,                      //!< current Newton-Raphson iteration step
        const int& itemax,                     //!< maximum number of Newton-Raphson iteration steps
        const double& ittol,                   //!< relative tolerance for Newton-Raphson scheme
        const std::vector<double>& preresnorm  //!< L2 norm of pressure residual
    );

    //! print current line of convergence table to screen
    virtual void print_convergence_values(
        const int& itnum,     //!< current Newton-Raphson iteration step
        const int& itemax,    //!< maximum number of Newton-Raphson iteration steps
        const double& ittol,  //!< relative tolerance for Newton-Raphson scheme
        const std::vector<double>& preresnorm,  //!< norm of pressure residual
        const std::vector<double>& incprenorm,  //!< norm of pressure increment
        const std::vector<double>& prenorm      //!< norm of pressure state vector
    );

    //! print finish line of convergence table to screen
    virtual void print_convergence_finish_line();

    // return arterial network time integrator
    std::shared_ptr<Adapter::ArtNet> art_net_tim_int() override;

    /*========================================================================*/
    //! @name Time, time-step and related methods
    /*========================================================================*/

    /*--- set, prepare, and predict ------------------------------------------*/

    //! increment time and step value
    void increment_time_and_step();

    /*========================================================================*/
    //! @name general framework variables
    /*========================================================================*/

    //! linear solver
    std::shared_ptr<Core::LinAlg::Solver> solver_;

    //! solver number in input file
    const int linsolvernumber_;

    //! parameter list of global control problem
    const Teuchos::ParameterList& params_;

    //! parameter list of poro fluid multiphase problem
    const Teuchos::ParameterList& poroparams_;

    //! processor id
    int myrank_;

    //! number of space dimensions
    int nsd_;

    /*========================================================================*/
    //! @name flags and enums
    /*========================================================================*/

    //! flag for Eulerian or ALE formulation of equation(s)
    bool isale_;

    //! flag if initial time derivative should be skipped
    bool skipinitder_;

    //! flag if saturations and pressures should be output
    bool output_satpress_;

    //! flag if solid pressure should be output
    bool output_solidpress_;

    //! flag if porosity should be output
    bool output_porosity_;

    //! flag if volfrac blood lung should be output
    bool output_volfrac_blood_lung_;

    //! flag if determinant of deformation gradient should be output
    bool output_det_def_grad_;

    //! flag if phase velocities should be written to output
    bool output_phase_velocities_;

    //! flag if blood vessel volume fraction should be output (for 1D-3D coupling)
    bool output_bloodvesselvolfrac_;

    //! flag for biot stabilization
    bool stab_biot_;

    /*--- query and output ---------------------------------------------------*/

    //! parameters for domain integration
    std::vector<int> domainint_funct_;
    int num_domainint_funct_;

    //! values of domain integrals
    std::shared_ptr<Core::LinAlg::SerialDenseVector> domain_integrals_;

    //! flag for error calculation
    const bool calcerr_;

    //! flag for flux reconstruction
    const bool flux_reconstruction_active_;

    //! solver number for flux reconstruction
    const int fluxreconsolvernum_;

    //! what to do when nonlinear solution fails
    enum PoroPressureBased::DivergenceAction divcontype_;

    //! flag for finite difference check
    const bool fdcheck_;

    //! perturbation magnitude for finite difference check
    const double fdcheckeps_;

    //! relative tolerance for finite difference check
    const double fdchecktol_;

    //! scaling factor for biot stabilization
    double stab_biot_scaling_;

    /*========================================================================*/
    //! @name Time, time-step, and iteration variables
    /*========================================================================*/

    //! actual time
    double time_;

    //! maximum simulation time
    double maxtime_;

    //! actual step number
    int step_;

    //! maximum number of steps
    const int stepmax_;

    //! time step size
    double dt_;

    //! time measurement element
    double dtele_;

    //! time measurement solve
    double dtsolve_;

    //! number of newton iterations in actual timestep
    int iternum_;

    //! maximum number of newton iterations
    const int itemax_;

    //! write results every upres_ steps ? writesolutionevery_
    const int upres_;

    //! write restart data every uprestart_ steps ? writesolutioneveryrestart_
    const int uprestart_;

    // vector norm for residuals
    enum PoroPressureBased::VectorNorm vectornormfres_;
    // vector norm for increments
    enum PoroPressureBased::VectorNorm vectornorminc_;

    //! convergence tolerance for increments
    double ittolres_;
    //! convergence tolerance for residuals
    double ittolinc_;

    //! flag if artery coupling is active
    bool artery_coupling_active_;

    /*========================================================================*/
    //! @name degrees of freedom variables
    /*========================================================================*/

    //! phi at time n
    std::shared_ptr<Core::LinAlg::Vector<double>> phin_;
    //! phi at time n+1
    std::shared_ptr<Core::LinAlg::Vector<double>> phinp_;

    //! time derivative of phi at time n
    std::shared_ptr<Core::LinAlg::Vector<double>> phidtn_;
    //! time derivative of phi at time n+1
    std::shared_ptr<Core::LinAlg::Vector<double>> phidtnp_;

    //! histvector --- a linear combination of phinm, phin (BDF)
    //!                or phin, phidtn (One-Step-Theta)
    std::shared_ptr<Core::LinAlg::Vector<double>> hist_;

    /*========================================================================*/
    //! @name degrees of freedom and related
    /*========================================================================*/

    //! pressure at time n+1
    std::shared_ptr<Core::LinAlg::Vector<double>> pressure_;

    //! saturation at time n+1
    std::shared_ptr<Core::LinAlg::Vector<double>> saturation_;

    //! solid pressure at time n+1
    std::shared_ptr<Core::LinAlg::Vector<double>> solidpressure_;

    //! porosity at time n+1
    std::shared_ptr<Core::LinAlg::Vector<double>> porosity_;

    //! volfrac of additional porous network with closing relation blood lung at time n+1
    std::shared_ptr<Core::LinAlg::Vector<double>> volfrac_blood_lung_;

    //! determinant of deformation gradient at time n+1
    std::shared_ptr<Core::LinAlg::Vector<double>> det_def_grad_;

    //! vector with valid volume fraction pressure dofs, this vector identifies volume fraction
    //! pressure DOFs,
    //  which actually have to be evaluated with a double >= 1.0, see also
    //  EvaluatorValidVolFracPressuresHomogenizedVasculatureTumor: if at least one nodal volume
    //  fraction value of an element is bigger than a threshold (min volfrac), the volume fraction
    //  pressure is a valid (physically meaningful) quantity in this element and the respective
    //  Darcy equation has to be solved for volume fraction species we only evaluate if all nodal
    //  volume fraction values of the element are bigger than the threshold (min volfrac), this
    //  turned out to be the most stable approach
    std::shared_ptr<Core::LinAlg::Vector<double>> valid_volfracpress_dofs_;
    std::shared_ptr<Core::LinAlg::Vector<double>> valid_volfracspec_dofs_;

    //! flux of each phase at time n+1 (post-processed from pressure solution)
    std::shared_ptr<Core::LinAlg::MultiVector<double>> flux_;

    //! velocity of each phase at time n+1 (post-processed from pressure solution)
    std::shared_ptr<Core::LinAlg::MultiVector<double>> phase_velocities_;

    //! number of dofset associated with displacement dofs
    int nds_disp_;

    //! number of dofset associated with velocity related dofs
    int nds_vel_;

    //! number of dofset associated with solid pressure dofs
    int nds_solidpressure_;

    //! number of dofset associated with scatra dofs
    int nds_scatra_;

    /*========================================================================*/
    //! @name Galerkin discretization, boundary conditions, and related
    /*========================================================================*/

    //! the porous multiphase flow discretization
    std::shared_ptr<Core::FE::Discretization> discret_;

    //! the discretization writer
    std::shared_ptr<Core::IO::DiscretizationWriter> output_;

    //! system matrix (either sparse matrix or block sparse matrix)
    std::shared_ptr<Core::LinAlg::SparseOperator> sysmat_;

    //! a vector of zeros to be used to enforce zero dirichlet boundary conditions
    std::shared_ptr<Core::LinAlg::Vector<double>> zeros_;

    //! maps for extracting Dirichlet and free DOF sets
    std::shared_ptr<Core::LinAlg::MapExtractor> dbcmaps_;

    //! maps for extracting Dirichlet and free DOF sets, here the additional dofs have been added
    //! which have to be zeroed out for the volume fraction pressure since it is not defined if the
    //! corresponding volume fraction is equal to zero (or smaller than minvolfrac)
    std::shared_ptr<Core::LinAlg::MapExtractor> dbcmaps_with_volfracpress_;

    //! maps for extracting Dirichlet and free DOF sets with additional starting Dirichlet boundary
    //! condition
    std::shared_ptr<Core::LinAlg::MapExtractor> dbcmaps_starting_condition_;

    //! the vector containing body and surface forces
    std::shared_ptr<Core::LinAlg::Vector<double>> neumann_loads_;

    //! residual vector
    std::shared_ptr<Core::LinAlg::Vector<double>> residual_;

    //! true (rescaled) residual vector without zeros at Dirichlet conditions
    std::shared_ptr<Core::LinAlg::Vector<double>> trueresidual_;

    //! nonlinear iteration increment vector
    std::shared_ptr<Core::LinAlg::Vector<double>> increment_;

    //! meshtying strategy
    std::shared_ptr<MeshtyingArtery> meshtying_;

    //! end time point when to switch off the starting Dirichlet boundary condition
    double starting_dbc_time_end_;

    //! switch the starting Dirichlet boundary condition on or off for the different DOFs
    std::vector<bool> starting_dbc_onoff_;

    //! function ID prescribing the starting Dirichlet boundary condition
    std::vector<int> starting_dbc_funct_;

    //! time factor for one-step-theta time integration
    double theta_;

    //! pointer to visualization writer object
    std::unique_ptr<Core::IO::DiscretizationVisualizationWriterMesh> visualization_writer_;

    /*========================================================================*/

  };  // class Porofluid
}  // namespace PoroPressureBased


FOUR_C_NAMESPACE_CLOSE

#endif
