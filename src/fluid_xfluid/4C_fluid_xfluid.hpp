// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FLUID_XFLUID_HPP
#define FOUR_C_FLUID_XFLUID_HPP

/*==========================================================================*/
// Style guide                                                    nis Mar12
/*==========================================================================*/

/*--- set, prepare, and predict ------------------------------------------*/

/*--- calculate and update -----------------------------------------------*/

/*--- query and output ---------------------------------------------------*/



/*==========================================================================*/
// header inclusions
/*==========================================================================*/


#include "4C_config.hpp"

#include "4C_cut_input.hpp"
#include "4C_fluid_implicit_integration.hpp"
#include "4C_fluid_xfluid_state.hpp"
#include "4C_xfem_input.hpp"

FOUR_C_NAMESPACE_OPEN

/*==========================================================================*/
// forward declarations
/*==========================================================================*/

namespace Core::FE
{
  class Discretization;
  class IndependentDofSet;
}  // namespace Core::FE
namespace Core::LinAlg
{
  class Solver;
  class SparseMatrix;
  class MultiMapExtractor;
  class MapExtractor;
  class BlockSparseMatrixBase;
  class SparseOperator;
}  // namespace Core::LinAlg


namespace Cut
{
  class CutWizard;
  class ElementHandle;
  class VolumeCell;
}  // namespace Cut


namespace Core::IO
{
  class DiscretizationWriter;
}

namespace XFEM
{
  class ConditionManager;
  class DiscretizationXFEM;
  class MeshCoupling;
  class XFEMDofSet;
  class XfemEdgeStab;
  class XFluidTimeInt;
}  // namespace XFEM


namespace FLD
{
  namespace Utils
  {
    class FluidInfNormScaling;
  }

  class XFluidStateCreator;
  class XFluidState;
  class XFluidOutputService;

  /*!
    This class holds the Fluid implementation for XFEM

  */
  class XFluid : public FluidImplicitTimeInt
  {
    friend class XFluidResultTest;

   public:
    /// Constructor
    XFluid(const std::shared_ptr<Core::FE::Discretization>&
               actdis,  ///< background fluid discretization
        const std::shared_ptr<Core::FE::Discretization>& mesh_coupdis,
        const std::shared_ptr<Core::FE::Discretization>& levelset_coupdis,
        const std::shared_ptr<Core::LinAlg::Solver>& solver,    ///< fluid solver
        const std::shared_ptr<Teuchos::ParameterList>& params,  ///< xfluid params
        const std::shared_ptr<Core::IO::DiscretizationWriter>&
            output,            ///< discretization writer for paraview output
        bool alefluid = false  ///< flag for alefluid
    );

    static void setup_fluid_discretization();

    /// initialization
    void init() override { init(true); }
    virtual void init(bool createinitialstate);

    void add_additional_scalar_dofset_and_coupling();

    void check_initialized_dof_set_coupling_map();

    /// print information about current time step to screen
    void print_time_step_info() override;

    void integrate() override { time_loop(); }

    /// Do time integration (time loop)
    void time_loop() override;

    /// setup the variables to do a new time step
    void prepare_time_step() override;

    /// set theta for specific time integration scheme
    void set_theta() override;

    /// do explicit predictor step
    void do_predictor();

    /// Implement Adapter::Fluid
    virtual void prepare_xfem_solve();

    /// do nonlinear iteration, e.g. full Newton, Newton like or Fixpoint iteration
    void solve() override;

    /// compute lift and drag values by integrating the true residuals
    void lift_drag() const override;

    /// solve linearised fluid
    void linear_solve();

    /// create vectors for KrylovSpaceProjection
    void init_krylov_space_projection() override;
    void setup_krylov_space_projection(const Core::Conditions::Condition* kspcond) override;
    void update_krylov_space_projection() override;
    void check_matrix_nullspace() override;

    /// return Teuchos::rcp to linear solver
    std::shared_ptr<Core::LinAlg::Solver> linear_solver() override { return solver_; };

    /// evaluate errors compared to implemented analytical solutions
    std::shared_ptr<std::vector<double>> evaluate_error_compared_to_analytical_sol() override;

    /// update velnp by increments
    virtual void update_by_increments(std::shared_ptr<const Core::LinAlg::Vector<double>>
            stepinc  ///< solution increment between time step n and n+1
    );


    /// build linear system matrix and rhs
    /// Monolithic FSI needs to access the linear fluid problem.
    virtual void evaluate();

    /// Update the solution after convergence of the nonlinear
    /// iteration. Current solution becomes old solution of next timestep.
    void time_update() override;

    /// Implement Adapter::Fluid
    void update() override { time_update(); }

    /// CUT at new interface position, transform vectors,
    /// perform time integration and set new Vectors
    void cut_and_set_state_vectors();

    /// is a restart of the monolithic Newton necessary caused by changing dofsets?
    bool newton_restart_monolithic() { return newton_restart_monolithic_; }

    /// ...
    std::shared_ptr<std::map<int, int>> get_permutation_map() { return permutation_map_; }

    /// update configuration and output to file/screen
    void output() override;

    /// set an initial flow field
    void set_initial_flow_field(
        const Inpar::FLUID::InitialField initfield, const int startfuncno) override;

    //    /// compute interface velocities from function
    //    void ComputeInterfaceVelocities();

    /// set Dirichlet and Neumann boundary conditions
    void set_dirichlet_neumann_bc() override;


    //! @name access methods for composite algorithms
    /// monolithic FSI needs to access the linear fluid problem
    std::shared_ptr<const Core::LinAlg::Vector<double>> initial_guess() override
    {
      return state_->incvel_;
    }
    std::shared_ptr<Core::LinAlg::Vector<double>> residual() override { return state_->residual_; }
    /// implement adapter fluid
    std::shared_ptr<const Core::LinAlg::Vector<double>> rhs() override { return residual(); }
    std::shared_ptr<const Core::LinAlg::Vector<double>> true_residual() override
    {
      std::cout << "Xfluid_TrueResidual" << std::endl;
      return state_->trueresidual_;
    }
    std::shared_ptr<const Core::LinAlg::Vector<double>> velnp() override { return state_->velnp_; }
    std::shared_ptr<const Core::LinAlg::Vector<double>> velaf() override { return state_->velaf_; }
    std::shared_ptr<const Core::LinAlg::Vector<double>> veln() override { return state_->veln_; }
    /*!
    \brief get the velocity vector based on standard dofs

    \return std::shared_ptr to a copy of Velnp with only standard dofs
     */
    std::shared_ptr<Core::LinAlg::Vector<double>> std_velnp() override;
    std::shared_ptr<Core::LinAlg::Vector<double>> std_veln() override;

    std::shared_ptr<const Core::LinAlg::Vector<double>> grid_vel() override
    {
      return gridvnp_;
    }  // full grid velocity (1st dofset)

    std::shared_ptr<const Core::LinAlg::Vector<double>> dispnp() override
    {
      return dispnp_;
    }  // full Dispnp (1st dofset)
    std::shared_ptr<Core::LinAlg::Vector<double>> write_access_dispnp() override
    {
      return dispnp_;
    }  // full Dispnp (1st dofset)
    std::shared_ptr<const Core::LinAlg::Vector<double>> dispn() override
    {
      return dispn_;
    }  // full Dispn(1st dofset)
    // @}


    std::shared_ptr<const Core::LinAlg::Map> dof_row_map() override
    {
      return state_->xfluiddofrowmap_;
    }

    std::shared_ptr<Core::LinAlg::MapExtractor> vel_pres_splitter() override
    {
      return state_->velpressplitter_;
    }
    std::shared_ptr<const Core::LinAlg::Map> velocity_row_map() override
    {
      return state_->velpressplitter_->other_map();
    }
    std::shared_ptr<const Core::LinAlg::Map> pressure_row_map() override
    {
      return state_->velpressplitter_->cond_map();
    }

    std::shared_ptr<Core::LinAlg::SparseMatrix> system_matrix() override
    {
      return std::dynamic_pointer_cast<Core::LinAlg::SparseMatrix>(state_->sysmat_);
    }
    std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> block_system_matrix() override
    {
      return std::dynamic_pointer_cast<Core::LinAlg::BlockSparseMatrixBase>(state_->sysmat_);
    }

    /// return coupling matrix between fluid and structure as sparse matrices
    std::shared_ptr<Core::LinAlg::SparseMatrix> c_sx_matrix(const std::string& cond_name);
    std::shared_ptr<Core::LinAlg::SparseMatrix> c_xs_matrix(const std::string& cond_name);
    std::shared_ptr<Core::LinAlg::SparseMatrix> c_ss_matrix(const std::string& cond_name);
    std::shared_ptr<Core::LinAlg::Vector<double>> rhs_s_vec(const std::string& cond_name);

    /// Return MapExtractor for Dirichlet boundary conditions
    std::shared_ptr<const Core::LinAlg::MapExtractor> get_dbc_map_extractor() override
    {
      return state_->dbcmaps_;
    }

    /// set the maximal number of nonlinear steps
    void set_itemax(int itemax) override { params_->set<int>("max nonlin iter steps", itemax); }

    /// scale the residual (inverse of the weighting of the quantities w.r.t the new timestep)
    double residual_scaling() const override
    {
      if (tim_int_scheme() == Inpar::FLUID::timeint_stationary)
        return 1.0;
      else if (tim_int_scheme() == Inpar::FLUID::timeint_afgenalpha)
        return alphaM_ / (gamma_ * dta_);
      else
        return 1.0 / (theta_ * dta_);
    }

    /// return time integration factor
    double tim_int_param() const override;

    /// turbulence statistics manager
    std::shared_ptr<FLD::TurbulenceStatisticManager> turbulence_statistic_manager() override
    {
      return nullptr;
    }

    /// create field test
    std::shared_ptr<Core::Utils::ResultTest> create_field_test() override;

    /// read restart data for fluid discretization
    void read_restart(int step) override;


    // -------------------------------------------------------------------
    std::shared_ptr<XFEM::MeshCoupling> get_mesh_coupling(const std::string& condname);


    std::shared_ptr<FLD::DynSmagFilter> dyn_smag_filter() override { return nullptr; }

    std::shared_ptr<FLD::Vreman> vreman() override { return nullptr; }

    /*!
    \brief velocity required for evaluation of related quantities required on element level

    */
    std::shared_ptr<const Core::LinAlg::Vector<double>> evaluation_vel() override
    {
      return nullptr;
    }


    virtual void create_initial_state();

    virtual void update_ale_state_vectors(std::shared_ptr<FLD::XFluidState> state = nullptr);

    void update_gridv() override;

    /// Get xFluid Background discretization
    std::shared_ptr<XFEM::DiscretizationXFEM> discretisation_xfem() { return xdiscret_; }

    /// Get XFEM Condition Manager
    std::shared_ptr<XFEM::ConditionManager> get_condition_manager() { return condition_manager_; }

    /// evaluate the CUT for in the next fluid evaluate
    void set_evaluate_cut(bool evaluate_cut) { evaluate_cut_ = evaluate_cut; }

    /// Get Cut Wizard
    std::shared_ptr<Cut::CutWizard> get_cut_wizard()
    {
      if (state_ != nullptr)
        return state_->wizard();
      else
        return nullptr;
    }

    /// Get xFluid ParameterList
    std::shared_ptr<Teuchos::ParameterList> params() { return params_; }

    /// Set state vectors depending on time integration scheme
    void set_state_tim_int() override;

   protected:
    /// (pseudo-)timeloop finished?
    bool not_finished() override;

    /// create a new state class object
    virtual void create_state();

    /// destroy state class' data (free memory of matrices, vectors ... )
    virtual void destroy_state();

    /// get a new state class
    virtual std::shared_ptr<FLD::XFluidState> get_new_state();

    void extract_node_vectors(XFEM::DiscretizationXFEM& dis,
        std::map<int, Core::LinAlg::Matrix<3, 1>>& nodevecmap,
        Core::LinAlg::Vector<double>& dispnp_col);

    /// call the loop over elements to assemble volume and interface integrals
    virtual void assemble_mat_and_rhs(int itnum  ///< iteration number
    );

    /// evaluate and assemble volume integral based terms
    void assemble_mat_and_rhs_vol_terms();

    /// evaluate and assemble face-oriented fluid and ghost penalty stabilizations
    void assemble_mat_and_rhs_face_terms(const std::shared_ptr<Core::LinAlg::SparseMatrix>& sysmat,
        const std::shared_ptr<Core::LinAlg::Vector<double>>& residual_col, Cut::CutWizard& wizard,
        bool is_ghost_penalty_reconstruct = false);

    /// evaluate gradient penalty terms to reconstruct ghost values
    void assemble_mat_and_rhs_gradient_penalty(Core::LinAlg::MapExtractor& ghost_penalty_dbcmaps,
        std::shared_ptr<Core::LinAlg::SparseMatrix> sysmat_gp,
        Core::LinAlg::Vector<double>& residual_gp,
        std::shared_ptr<Core::LinAlg::Vector<double>> vec);

    /// integrate the shape function and assemble into a vector for KrylovSpaceProjection
    void integrate_shape_function(Teuchos::ParameterList& eleparams,  ///< element parameters
        Core::FE::Discretization& discret,  ///< background fluid discretization
        Core::LinAlg::Vector<double>& vec   ///< vector into which we assemble
    );

    /*!
    \brief convergence check

    */
    bool convergence_check(int itnum, int itemax, const double velrestol, const double velinctol,
        const double presrestol, const double presinctol) override;

    /// Update velocity and pressure by increment
    virtual void update_by_increment();

    void set_old_part_of_righthandside() override;

    void set_old_part_of_righthandside(Core::LinAlg::Vector<double>& veln,
        Core::LinAlg::Vector<double>& velnm, Core::LinAlg::Vector<double>& accn,
        const Inpar::FLUID::TimeIntegrationScheme timealgo, const double dta, const double theta,
        Core::LinAlg::Vector<double>& hist);

    void set_gamma(Teuchos::ParameterList& eleparams) override;

    /*!
    \brief Scale separation

    */
    void sep_multiply() override { return; }

    void outputof_filtered_vel(std::shared_ptr<Core::LinAlg::Vector<double>> outvec,
        std::shared_ptr<Core::LinAlg::Vector<double>> fsoutvec) override
    {
      return;
    }

    /*!
  \brief Calculate time derivatives for
         stationary/one-step-theta/BDF2/af-generalized-alpha time integration
         for incompressible and low-Mach-number flow
     */
    void calculate_acceleration(
        const std::shared_ptr<const Core::LinAlg::Vector<double>> velnp,  ///< velocity at n+1
        const std::shared_ptr<const Core::LinAlg::Vector<double>> veln,   ///< velocity at     n
        const std::shared_ptr<const Core::LinAlg::Vector<double>> velnm,  ///< velocity at     n-1
        const std::shared_ptr<const Core::LinAlg::Vector<double>> accn,   ///< acceleration at n-1
        const std::shared_ptr<Core::LinAlg::Vector<double>> accnp         ///< acceleration at n+1
        ) override;

    //-----------------------------XFEM time-integration specific function------------------

    //! @name XFEM time-integration specific function

    /// store state data from old time-step t^n
    virtual void x_timint_store_old_state_data(const bool firstcall_in_timestep);

    /// is a restart of the global monolithic system necessary?
    bool x_timint_check_for_monolithic_newton_restart(
        const bool timint_ghost_penalty,    ///< dofs have to be reconstructed via ghost penalty
                                            ///< reconstruction techniques
        const bool timint_semi_lagrangean,  ///< dofs have to be reconstructed via semi-Lagrangean
                                            ///< reconstruction techniques
        Core::FE::Discretization& dis,      ///< discretization
        XFEM::XFEMDofSet& dofset_i,         ///< dofset last iteration
        XFEM::XFEMDofSet& dofset_ip,        ///< dofset current iteration
        const bool screen_out               ///< screen output?
    );

    /// Transfer vectors from old time-step t^n w.r.t dofset and interface position
    /// from t^n to vectors w.r.t current dofset and interface position
    virtual void x_timint_do_time_step_transfer(const bool screen_out);

    /// Transfer vectors at current time-step t^(n+1) w.r.t dofset and interface position
    /// from last iteration i to vectors w.r.t current dofset and interface position (i+1)
    ///
    /// return, if increment step transfer was successful!
    virtual bool x_timint_do_increment_step_transfer(
        const bool screen_out, const bool firstcall_in_timestep);

    /// did the dofsets change?
    bool x_timint_changed_dofsets(Core::FE::Discretization& dis,  ///< discretization
        XFEM::XFEMDofSet& dofset,                                 ///< first dofset
        XFEM::XFEMDofSet& dofset_other                            ///< other dofset
    );

    /// transfer vectors between two time-steps or Newton steps
    void x_timint_transfer_vectors_between_steps(
        const std::shared_ptr<XFEM::XFluidTimeInt>&
            xfluid_timeint,  ///< xfluid time integration class
        std::vector<std::shared_ptr<const Core::LinAlg::Vector<double>>>&
            oldRowStateVectors,  ///< row map based vectors w.r.t old interface position
        std::vector<std::shared_ptr<Core::LinAlg::Vector<double>>>&
            newRowStateVectors,  ///< row map based vectors w.r.t new interface position
        std::shared_ptr<std::set<int>> dbcgids,  ///< set of dof gids that must not be changed by
                                                 ///< ghost penalty reconstruction
        bool fill_permutation_map,
        bool screen_out  ///< output to screen
    );

    void x_timint_corrective_transfer_vectors_between_steps(
        const std::shared_ptr<XFEM::XFluidTimeInt>&
            xfluid_timeint,                                     ///< xfluid time integration class
        const XFEM::XFluidTimeIntScheme xfluid_timintapproach,  /// xfluid_timintapproach
        std::vector<std::shared_ptr<const Core::LinAlg::Vector<double>>>&
            oldRowStateVectors,  ///< row map based vectors w.r.t old interface position
        std::vector<std::shared_ptr<Core::LinAlg::Vector<double>>>&
            newRowStateVectors,  ///< row map based vectors w.r.t new interface position
        std::shared_ptr<std::set<int>> dbcgids,  ///< set of dof gids that must not be changed by
                                                 ///< ghost penalty reconstruction
        bool screen_out                          ///< output to screen
    );

    /// decide if semi-Lagrangean back-tracking or ghost-penalty reconstruction has to be
    /// performed on any processor
    void x_timint_get_reconstruct_status(const std::shared_ptr<XFEM::XFluidTimeInt>&
                                             xfluid_timeint,  ///< xfluid time integration class
        bool& timint_ghost_penalty,   ///< do we have to perform ghost penalty reconstruction of
                                      ///< ghost values?
        bool& timint_semi_lagrangean  ///< do we have to perform semi-Lagrangean reconstruction of
                                      ///< standard values?
    );

    /// create DBC and free map and return their common extractor
    std::shared_ptr<Core::LinAlg::MapExtractor> create_dbc_map_extractor(
        const std::set<int>& dbcgids,       ///< dbc global dof ids
        const Core::LinAlg::Map* dofrowmap  ///< dofrowmap
    );

    /// create new dbc maps for ghost penalty reconstruction and reconstruct value which are not
    /// fixed by DBCs
    void x_timint_ghost_penalty(std::vector<std::shared_ptr<Core::LinAlg::Vector<double>>>&
                                    rowVectors,  ///< vectors to be reconstructed
        const Core::LinAlg::Map* dofrowmap,      ///< dofrowmap
        const std::set<int>& dbcgids,            ///< dbc global ids
        const bool screen_out                    ///< screen output?
    );

    /// reconstruct ghost values using ghost penalty approach
    void x_timint_reconstruct_ghost_values(
        std::shared_ptr<Core::LinAlg::Vector<double>> vec,  ///< vector to be reconstructed
        Core::LinAlg::MapExtractor& ghost_penalty_dbcmaps,  ///< which dofs are fixed during the
                                                            ///< ghost-penalty reconstruction?
        const bool screen_out                               ///< screen output?
    );

    /// reconstruct standard values using semi-Lagrangean method
    void x_timint_semi_lagrangean(std::vector<std::shared_ptr<Core::LinAlg::Vector<double>>>&
                                      newRowStateVectors,  ///< vectors to be reconstructed
        const Core::LinAlg::Map* newdofrowmap,  ///< dofrowmap at current interface position
        std::vector<std::shared_ptr<const Core::LinAlg::Vector<double>>>&
            oldRowStateVectors,  ///< vectors from which we reconstruct values (same order of
                                 ///< vectors as in newRowStateVectors)
        std::shared_ptr<Core::LinAlg::Vector<double>>
            dispn,  ///< displacement col - vector timestep n
        std::shared_ptr<Core::LinAlg::Vector<double>>
            dispnp,                             ///< displacement col - vector timestep n+1
        const Core::LinAlg::Map* olddofcolmap,  ///< dofcolmap at time and interface position t^n
        std::map<int, std::vector<XFEM::XFluidTimeIntMethod>>&
            node_to_reconstr_method,  ///< reconstruction map for nodes and its dofsets
        const bool screen_out         ///< screen output?
    );

    /// projection of history from other discretization - returns true if projection was
    /// successful for all nodes
    virtual bool x_timint_project_from_embedded_discretization(
        const std::shared_ptr<XFEM::XFluidTimeInt>&
            xfluid_timeint,  ///< xfluid time integration class
        std::vector<std::shared_ptr<Core::LinAlg::Vector<double>>>&
            newRowStateVectors,  ///< vectors to be reconstructed
        std::shared_ptr<const Core::LinAlg::Vector<double>>
            target_dispnp,     ///< displacement col - vector timestep n+1
        const bool screen_out  ///< screen output?
    )
    {
      return true;
    };

    //@}


    /// set xfluid input parameters (read from list)
    void set_x_fluid_params();

    /// check xfluid input parameters for consistency
    void check_x_fluid_params() const;

    /// print stabilization params to screen
    void print_stabilization_details() const override;

    //! @name Set general xfem specific element parameter in class FluidEleParameterXFEM
    /*!

    \brief parameter (fix over all time step) are set in this method.
    Therefore, these parameter are accessible in the fluid element
    and in the fluid boundary element

    */
    void set_element_general_fluid_xfem_parameter();

    //! @name Set general parameter in class f3Parameter
    /*!

    \brief parameter (fix over a time step) are set in this method.
    Therefore, these parameter are accessible in the fluid element
    and in the fluid boundary element

    */
    void set_element_time_parameter() override;

    //! @name Set general parameter in parameter list class for fluid internal face elements
    /*!

    \brief parameter (fix over all time step) are set in this method.
           Therefore, these parameter are accessible in the fluid intfaces element

    */
    void set_face_general_fluid_xfem_parameter();

    /// initialize vectors and flags for turbulence approach
    void set_general_turbulence_parameters() override;

    void explicit_predictor() override;

    void predict_tang_vel_consist_acc() override;

    void update_iter_incrementally(
        std::shared_ptr<const Core::LinAlg::Vector<double>> vel) override;

    void compute_error_norms(Core::LinAlg::SerialDenseVector& glob_dom_norms,
        Core::LinAlg::SerialDenseVector& glob_interf_norms,
        Core::LinAlg::SerialDenseVector& glob_stab_norms);

    /*!
      \brief compute values at intermediate time steps for gen.-alpha

    */
    void gen_alpha_intermediate_values() override;

    /*!
      \brief call elements to calculate system matrix/rhs and assemble

    */
    void assemble_mat_and_rhs() override;

    /*!
      \brief update acceleration for generalized-alpha time integration

    */
    void gen_alpha_update_acceleration() override;

    /// return type of enforcing interface conditions
    XFEM::CouplingMethod coupling_method() const { return coupling_method_; }

    //@}

    //    //! @name Get material properties for the Volume Cell
    //    /*!
    //
    //    \brief Element material for the volume cell, depending on element and position.
    //           If an element which is not a material list is given, the provided material is
    //           chosen. If however a material list is given the material chosen for the volume
    //           cell is depending on the point position.
    //
    //    */
    //    void get_volume_cell_material(Core::Elements::Element* actele,
    //                               std::shared_ptr<Core::Mat::Material> & mat,
    //                               const Cut::Point::PointPosition position =
    //                               Cut::Point::outside);


    //-------------------------------------------------------------------------------
    //! possible inf-norm scaling of linear system / fluid matrix
    std::shared_ptr<FLD::Utils::FluidInfNormScaling> fluid_infnormscaling_;

    //--------------- discretization and general algorithm parameters----------------

    //! @name discretizations

    //! xfem fluid discretization
    std::shared_ptr<XFEM::DiscretizationXFEM> xdiscret_;

    //! vector of all coupling discretizations, the fluid is coupled with
    std::vector<std::shared_ptr<Core::FE::Discretization>> meshcoupl_dis_;

    //! vector of all coupling discretizations, which carry levelset fields, the fluid is coupled
    //! with
    std::vector<std::shared_ptr<Core::FE::Discretization>> levelsetcoupl_dis_;

    //@}


    //---------------------------------input parameters------------------

    /// type of enforcing interface conditions in XFEM
    XFEM::CouplingMethod coupling_method_;

    //! @name xfluid time integration
    XFEM::XFluidTimeIntScheme xfluid_timintapproach_;

    //! @name check interfacetips in timeintegration
    bool xfluid_timint_check_interfacetips_;

    //! @name check sliding on surface in timeintegration
    bool xfluid_timint_check_sliding_on_surface_;
    //@}

    /// initial flow field
    Inpar::FLUID::InitialField initfield_;

    /// start function number for an initial field
    int startfuncno_;


    //@}

    //! @name

    /// edge stabilization and ghost penalty object
    std::shared_ptr<XFEM::XfemEdgeStab> edgestab_;

    /// edgebased stabilization or ghost penalty stabilization (1st order, 2nd order derivatives,
    /// viscous, transient) due to Nitsche's method
    bool eval_eos_;

    //---------------------------------output----------------------------

    // counter for current Newton iteration (used for Debug output)
    int itnum_out_;

    /// vel-pres splitter for output purpose (and outer iteration convergence)
    std::shared_ptr<Core::LinAlg::MapExtractor> velpressplitter_std_;

    /// output service class
    std::shared_ptr<XFluidOutputService> output_service_;
    //--------------------------------------------------------------------

    //! do we have a turblence model?
    Inpar::FLUID::TurbModelAction turbmodel_;
    //@}


    /// number of spatial dimensions
    int numdim_;

    //! @name time stepping variables
    bool startalgo_;  ///< flag for starting algorithm
    //@}

    /// constant density extracted from element material for incompressible flow
    /// (set to 1.0 for low-Mach-number flow)
    double density_;

    /// for low-Mach-number flow solver: thermodynamic pressure at n+alpha_F/n+1
    /// and at n+alpha_M/n as well as its time derivative at n+alpha_F/n+1 and n+alpha_M/n
    double thermpressaf_;
    double thermpressam_;
    double thermpressdtaf_;
    double thermpressdtam_;


    //! @name time-integration-scheme factors
    double omtheta_;
    double alphaM_;
    double alphaF_;
    double gamma_;
    //@}


    //--------------------------------------------------------------------

    /// state creator object
    std::shared_ptr<FLD::XFluidStateCreator> state_creator_;

    /// object to handle the different types of XFEM boundary and interface coupling conditions
    std::shared_ptr<XFEM::ConditionManager> condition_manager_;

    int mc_idx_;

    bool include_inner_;

    /// Apply ghost penalty stabilization also for inner faces when possible
    bool ghost_penalty_add_inner_faces_;

    //--------------------------------------------------------------------

   public:
    /// state object at current new time step
    std::shared_ptr<FLD::XFluidState> state_;

   protected:
    /// state object of previous time step
    std::shared_ptr<FLD::XFluidState> staten_;

    /// evaluate the CUT for in the next fluid evaluate
    bool evaluate_cut_;

    /// counter how often a state class has been created during one time-step
    int state_it_;


    //---------------------------------dofsets----------------------------

    //! @name dofset variables for dofsets with variable size
    int maxnumdofsets_;
    int minnumdofsets_;
    //@}

    //------------------------------- vectors -----------------------------

    //! @name full fluid-field vectors

    // Dispnp of full fluidfield (also unphysical area - to avoid reconstruction for gridvelocity
    // calculation!)
    std::shared_ptr<Core::LinAlg::Vector<double>> dispnp_;
    std::shared_ptr<Core::LinAlg::Vector<double>> dispn_;
    std::shared_ptr<Core::LinAlg::Vector<double>> dispnm_;

    /// grid velocity (set from the adapter!)
    std::shared_ptr<Core::LinAlg::Vector<double>> gridvnp_;

    /// grid velocity at timestep n
    std::shared_ptr<Core::LinAlg::Vector<double>> gridvn_;

    //@}


    //-----------------------------XFEM time-integration specific data ----------------

    //! @name old time-step state data w.r.t old interface position and dofsets from t^n used for
    //! XFEM time-integration
    std::shared_ptr<Core::LinAlg::Vector<double>>
        veln_Intn_;  //!< velocity solution from last time-step t^n
    std::shared_ptr<Core::LinAlg::Vector<double>>
        accn_Intn_;  //!< acceleration from last time-step t^n
    std::shared_ptr<Core::LinAlg::Vector<double>>
        velnm_Intn_;                                 //!< velocity at t^{n-1} for BDF2 scheme
    std::shared_ptr<Cut::CutWizard> wizard_Intn_;    //!< cut wizard from last time-step t^n
    std::shared_ptr<XFEM::XFEMDofSet> dofset_Intn_;  //!< dofset from last time-step t^n

    std::shared_ptr<Core::LinAlg::Map> dofcolmap_Intn_;
    //@}


    //! @name last iteration step state data from t^(n+1) used for pseudo XFEM time-integration
    //! during monolithic Newton or partitioned schemes
    std::shared_ptr<Core::LinAlg::Vector<double>>
        velnp_Intnpi_;  //!< velocity solution from last iteration
                        //!< w.r.t last dofset and interface position
    std::shared_ptr<Cut::CutWizard>
        wizard_Intnpi_;  //!< cut wizard from last iteration-step t^(n+1)
    std::shared_ptr<XFEM::XFEMDofSet> dofset_Intnpi_;  //!< dofset from last iteration-step t^(n+1)

    //! is a restart of the monolithic Newton necessary caused by changing dofsets?
    bool newton_restart_monolithic_;

    //! how did std/ghost dofs of nodes permute between the last two iterations
    std::shared_ptr<std::map<int, int>> permutation_map_;
    //@}

    std::map<std::string, int> dofset_coupling_map_;
  };
}  // namespace FLD

FOUR_C_NAMESPACE_CLOSE

#endif
