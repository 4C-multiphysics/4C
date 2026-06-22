// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FLUID_IMPLICIT_INTEGRATION_HPP
#define FOUR_C_FLUID_IMPLICIT_INTEGRATION_HPP


#include "4C_config.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_fluid_timint.hpp"
#include "4C_fluid_turbulence_input.hpp"
#include "4C_inpar_fluid.hpp"
#include "4C_linalg_blocksparsematrix.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

#include <Teuchos_TimeMonitor.hpp>

#include <ctime>
#include <iostream>

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Core::FE
{
  class Discretization;
  class DiscretizationFaces;
}  // namespace Core::FE
namespace Core::IO
{
  class DiscretizationWriter;
}
namespace Core::LinAlg
{
  class Solver;
  class SparseMatrix;
  class MultiMapExtractor;
  class MapExtractor;
  class BlockSparseMatrixBase;
  class SparseOperator;
}  // namespace Core::LinAlg

namespace Core::Conditions
{
  class LocsysManager;
}


namespace Adapter
{
  class CouplingMortar;
}

namespace FLD
{
  // forward declarations
  class TurbulenceStatisticManager;
  class ForcingInterface;
  class DynSmagFilter;
  class Vreman;
  class Boxfilter;
  class Meshtying;
  class XWall;
  class TransferTurbulentInflowCondition;
  namespace Utils
  {
    class FluidInfNormScaling;
    class FluidImpedanceWrapper;
    class StressManager;
  }  // namespace Utils

  /*!
  \brief time integration for fluid problems

  */
  class FluidImplicitTimeInt : public TimInt
  {
    friend class TurbulenceStatisticManager;
    friend class HomoIsoTurbInitialField;
    friend class HomoIsoTurbForcing;
    friend class PeriodicHillForcing;
    friend class FluidResultTest;

   public:
    /*!
    \brief Standard Constructor

    */
    FluidImplicitTimeInt(const std::shared_ptr<Core::FE::Discretization>& actdis,
        const std::shared_ptr<Core::LinAlg::Solver>& solver,
        const std::shared_ptr<Teuchos::ParameterList>& params,
        const std::shared_ptr<Core::IO::DiscretizationWriter>& output, bool alefluid = false);

    /*!
    \brief initialization

    */
    void init() override;

    /*!
    \brief initialization of nonlinear BCs

    */
    virtual void init_nonlinear_bc();

    /*!
    \brief start time loop for starting algorithm, normal problems and restarts

    */
    void integrate() override;

    /*!
    \brief Do time integration (time loop)

    */
    virtual void time_loop();

    /*!
    \brief Print information about current time step to screen

    */
    virtual void print_time_step_info()
    {
      FOUR_C_THROW("you are in the base class");
      return;
    }

    /*!
    \brief Set theta_ to its value, dependent on integration method for GenAlpha and BDF2

    */
    virtual void set_theta() { return; }

    /*!
    \brief Set the part of the righthandside belonging to the last
           timestep for incompressible or low-Mach-number flow

       for low-Mach-number flow: distinguish momentum and continuity part
       (continuity part only meaningful for low-Mach-number flow)

       Stationary/af-generalized-alpha:

                     mom: hist_ = 0.0
                    (con: hist_ = 0.0)

       One-step-Theta:

                     mom: hist_ = veln_  + dt*(1-Theta)*accn_
                    (con: hist_ = densn_ + dt*(1-Theta)*densdtn_)

       BDF2: for constant time step:

                     mom: hist_ = 4/3 veln_  - 1/3 velnm_
                    (con: hist_ = 4/3 densn_ - 1/3 densnm_)


    */
    virtual void set_old_part_of_righthandside() = 0;

    /*!
    \brief Set gamma to a value

    */
    virtual void set_gamma(Teuchos::ParameterList& eleparams) = 0;

    /*!
    \brief Initialize function which is called after that the constructor of the time integrator has
    been called

    */
    virtual void complete_general_init();

    /*!
     * \brief Create internal faces extension
     */
    void create_faces_extension();


    /*!
    \brief Set states in the time integration schemes: differs between GenAlpha and the others

    */
    virtual void set_state_tim_int() = 0;

    /*!
    \brief Set time factor in GenAlpha

    */
    virtual double set_time_fac() { return 1.0; }

    /*!
    \brief Scale separation

    */
    virtual void sep_multiply() = 0;

    /*!
    \brief Update velaf_ for GenAlpha

    */
    virtual void update_velaf_gen_alpha() {}

    /*!
    \brief Update velam_ for GenAlpha

    */
    virtual void update_velam_gen_alpha() {}

    /*!
    \brief Insert Womersley condition

    */
    virtual void insert_volumetric_surface_flow_cond_vector(
        std::shared_ptr<Core::LinAlg::Vector<double>> vel,
        std::shared_ptr<Core::LinAlg::Vector<double>> res)
    {
      return;
    }

    /*!
    \brief treat turbulence models in assemble_mat_and_rhs

    */
    virtual void treat_turbulence_models(Teuchos::ParameterList& eleparams);

    /*!
    \brief Evaluate for AVM3 Separation

    */
    virtual void avm3_assemble_mat_and_rhs(Teuchos::ParameterList& eleparams);

    /*!
    \brief Get scale separation matrix

    */
    virtual void avm3_get_scale_separation_matrix();

    /*!
    \brief Set custom parameters in the respective time integration class (Loma, RedModels...)

    */
    virtual void set_custom_ele_params_assemble_mat_and_rhs(Teuchos::ParameterList& eleparams) {}

    /*!
    \brief Call discret_->ClearState() after assembly (HDG needs to read from state vectors...)

    */
    virtual void clear_state_assemble_mat_and_rhs() { discret_->clear_state(); }

    /*!
    \brief Set custom parameters in the respective time integration class (Loma, RedModels...)

    */
    virtual void set_custom_ele_params_apply_nonlinear_boundary_conditions(
        Teuchos::ParameterList& eleparams)
    {
    }

    /*!
    \brief Set custom parameters in the respective time integration class (Loma, RedModels...)

    */
    virtual void set_custom_ele_params_linear_relaxation_solve(Teuchos::ParameterList& eleparams) {}

    /*!
    \brief Prepare calculation of acceleration

    */
    virtual void tim_int_calculate_acceleration();

    /*!
    \brief Additional function for RedModels in linear_relaxation_solve

    */
    virtual void custom_solve(std::shared_ptr<Core::LinAlg::Vector<double>> relax) {}

    /*!
    \brief Call statistics manager (special case in TimIntLoma)

    */
    virtual void call_statistics_manager();

    /*!
    \brief return thermpressaf_ in TimIntLoma

    */
    virtual double return_thermpressaf() { return 0.0; }

    /*!
    \brief Calculate time derivatives for
           stationary/one-step-theta/BDF2/af-generalized-alpha time integration
           for incompressible and low-Mach-number flow
    */
    virtual void calculate_acceleration(
        const std::shared_ptr<const Core::LinAlg::Vector<double>> velnp,  ///< velocity at     n+1
        const std::shared_ptr<const Core::LinAlg::Vector<double>> veln,   ///< velocity at     n
        const std::shared_ptr<const Core::LinAlg::Vector<double>> velnm,  ///< velocity at     n-1
        const std::shared_ptr<const Core::LinAlg::Vector<double>> accn,   ///< acceleration at n-1
        const std::shared_ptr<Core::LinAlg::Vector<double>> accnp         ///< acceleration at n+1
        ) = 0;

    //! @name Set general parameter in class f3Parameter
    /*!

    \brief parameter (fix over all time step) are set in this method.
           Therefore, these parameter are accessible in the fluid element
           and in the fluid boundary element

    */
    virtual void set_element_general_fluid_parameter();

    //! @name Set general parameter in class f3Parameter
    /*!

    \brief parameter (fix over all time step) are set in this method.
           Therefore, these parameter are accessible in the fluid element
           and in the fluid boundary element

    */
    virtual void set_element_turbulence_parameters();

    //! @name Set general parameter in class fluid_ele_parameter_intface
    /*!

    \brief parameter (fix over all time step) are set in this method.

    */
    virtual void set_face_general_fluid_parameter();

    /// initialize vectors and flags for turbulence approach
    virtual void set_general_turbulence_parameters();

    /*!
    \brief do explicit predictor step to start nonlinear iteration from
           a better initial value
                          +-                                      -+
                          | /     dta \          dta  veln_-velnm_ |
     velnp_ = veln_ + dta | | 1 + --- | accn_ - ----- ------------ |
                          | \     dtp /          dtp     dtp       |
                          +-                                      -+
    */
    virtual void explicit_predictor();

    /// setup the variables to do a new time step
    void prepare_time_step() override;

    /*!
    \brief (multiple) corrector

    */
    void solve() override;

    /*!
  \brief solve linearised fluid

  */
    std::shared_ptr<Core::LinAlg::Solver> linear_solver() override { return solver_; };

    /*!
    \brief preparatives for solver

    */
    void prepare_solve() override;

    /*!
    \brief preparations for Krylov space projection

    */
    virtual void init_krylov_space_projection();
    void setup_krylov_space_projection(const Core::Conditions::Condition* kspcond) override;
    void update_krylov_space_projection() override;
    void check_matrix_nullspace() override;

    /*!
    \brief update within iteration

    */
    void iter_update(const std::shared_ptr<const Core::LinAlg::Vector<double>> increment) override;

    /*!
   \brief convergence check

    */
    bool convergence_check(int itnum, int itmax, const double velrestol, const double velinctol,
        const double presrestol, const double presinctol) override;

    /*!
      \brief build linear system matrix and rhs

      Monolithic FSI needs to access the linear fluid problem.
    */
    void evaluate(std::shared_ptr<const Core::LinAlg::Vector<double>>
            stepinc  ///< solution increment between time step n and n+1
        ) override;


    /*!
    \brief Update the solution after convergence of the nonlinear
           iteration. Current solution becomes old solution of next
           timestep.
    */
    virtual void time_update();

    /*!
    \ time update of stresses
    */
    virtual void time_update_stresses();

    virtual void time_update_nonlinear_bc();

    /*!
    \ Update of external forces

    */
    virtual void time_update_external_forces();

    /// Implement Adapter::Fluid
    void update() override { time_update(); }

    //! @name Time step size adaptivity in monolithic FSI
    //@{

    //! access to time step size of previous time step
    virtual double dt_previous() const { return dtp_; }

    //! set time step size
    void set_dt(const double dtnew) override;

    //! set time and step
    void set_time_step(const double time,  ///< time to set
        const int step) override;          ///< time step number to set

    /*!
    \brief Reset time step

    In case of time step size adaptivity, time steps might have to be repeated.
    Therefore, we need to reset the solution back to the initial solution of the
    time step.

    */
    void reset_step() override
    {
      accnp_->update(1.0, *accn_, 0.0);
      velnp_->update(1.0, *veln_, 0.0);
      dispnp_->update(1.0, *dispn_, 0.0);

      return;
    }

    /*! \brief Reset time and step in case that a time step has to be repeated
     *
     *  Fluid field increments time and step at the beginning of a time step. If a
     *  time step has to be repeated, we need to take this into account and
     *  decrease time and step beforehand. They will be incremented right at the
     *  beginning of the repetition and, thus, everything will be fine.
     *
     *  Currently, this is needed for time step size adaptivity in FSI.
     *
     */
    void reset_time(const double dtold) override { set_time_step(time() - dtold, step() - 1); }

    //! Give order of accuracy
    virtual int method_order_of_accuracy() const
    {
      return std::min(method_order_of_accuracy_vel(), method_order_of_accuracy_pres());
    }

    //! Give local order of accuracy of velocity part
    virtual int method_order_of_accuracy_vel() const
    {
      FOUR_C_THROW("Not implemented in base class. May be overwritten by derived class.");
      return 0;
    }

    //! Give local order of accuracy of pressure part
    virtual int method_order_of_accuracy_pres() const
    {
      FOUR_C_THROW("Not implemented in base class. May be overwritten by derived class.");
      return 0;
    }
    //! Return linear error coefficient of velocity
    virtual double method_lin_err_coeff_vel() const
    {
      FOUR_C_THROW("Not implemented in base class. May be overwritten by derived class.");
      return 0;
    }

    //@}

    /*!
    \brief lift'n'drag forces, statistics time sample and
           output of solution and statistics

    */
    void statistics_and_output() override;

    /*!
    \brief statistics time sample and
           output of statistics

    */
    void statistics_output() override;

    /*!
    \brief update configuration and output to file/screen

    */
    void output() override;

    /*
     * \brief Write fluid runtime output
     */
    virtual void write_runtime_output();

    virtual void output_nonlinear_bc();

    virtual void output_to_gmsh(const int step, const double time, const bool inflow) const;

    /*!
    \output of external forces for restart

    */
    virtual void output_external_forces();

    /*!
    \brief get access to map extractor for velocity and pressure

    */
    std::shared_ptr<Core::LinAlg::MapExtractor> get_vel_press_splitter() override
    {
      return velpressplitter_;
    };

    /*!
    \brief set initial flow field for analytical test problems

    */
    void set_initial_flow_field(
        const Inpar::FLUID::InitialField initfield, const int startfuncno) override;

    /// Implement Adapter::Fluid
    std::shared_ptr<const Core::LinAlg::Vector<double>> extract_velocity_part(
        std::shared_ptr<const Core::LinAlg::Vector<double>> velpres) override;

    /// Implement Adapter::Fluid
    std::shared_ptr<const Core::LinAlg::Vector<double>> extract_pressure_part(
        std::shared_ptr<const Core::LinAlg::Vector<double>> velpres) override;

    /// Reset state vectors
    void reset(int numsteps = 1, int iter = -1) override;

    /*!
    \brief calculate error between a analytical solution and the
           numerical solution of a test problems

    */
    virtual std::shared_ptr<std::vector<double>> evaluate_error_compared_to_analytical_sol();

    /*!
    \brief evaluate divergence of velocity field

    */
    virtual std::shared_ptr<double> evaluate_div_u();

    /*!
    \brief calculate adaptive time step with the CFL number

    */
    virtual double evaluate_dt_via_cfl_if_applicable();

    /*!
    \brief read restart data

    */
    void read_restart(int step) override;

    /*!
    \brief get restart data in case of turbulent inflow computation

    */
    void set_restart(const int step, const double time,
        std::shared_ptr<const Core::LinAlg::Vector<double>> readvelnp,
        std::shared_ptr<const Core::LinAlg::Vector<double>> readveln,
        std::shared_ptr<const Core::LinAlg::Vector<double>> readvelnm,
        std::shared_ptr<const Core::LinAlg::Vector<double>> readaccnp,
        std::shared_ptr<const Core::LinAlg::Vector<double>> readaccn) override;

    //! @name access methods for composite algorithms
    /// monolithic FSI needs to access the linear fluid problem

    std::shared_ptr<const Core::LinAlg::Vector<double>> initial_guess() override { return incvel_; }

    /// return implemented residual (is not an actual force in Newton [N])
    virtual std::shared_ptr<Core::LinAlg::Vector<double>> residual() { return residual_; }

    /// implement adapter fluid
    std::shared_ptr<const Core::LinAlg::Vector<double>> rhs() override { return residual(); }

    /// Return true residual, ie the actual force in Newton [N]
    std::shared_ptr<const Core::LinAlg::Vector<double>> true_residual() override
    {
      return trueresidual_;
    }

    std::shared_ptr<const Core::LinAlg::Vector<double>> velnp() override { return velnp_; }
    std::shared_ptr<Core::LinAlg::Vector<double>> write_access_velnp() override { return velnp_; }
    std::shared_ptr<const Core::LinAlg::Vector<double>> velaf() override { return velaf_; }
    virtual std::shared_ptr<const Core::LinAlg::Vector<double>> velam() { return velam_; }
    std::shared_ptr<const Core::LinAlg::Vector<double>> veln() override { return veln_; }
    std::shared_ptr<const Core::LinAlg::Vector<double>> velnm() override { return velnm_; }
    virtual std::shared_ptr<Core::LinAlg::Vector<double>> write_access_accnp() { return accnp_; }
    std::shared_ptr<const Core::LinAlg::Vector<double>> accnp() override { return accnp_; }
    std::shared_ptr<const Core::LinAlg::Vector<double>> accn() override { return accn_; }
    std::shared_ptr<const Core::LinAlg::Vector<double>> accnm() override { return accnm_; }
    std::shared_ptr<const Core::LinAlg::Vector<double>> accam() override { return accam_; }
    std::shared_ptr<const Core::LinAlg::Vector<double>> scaaf() override { return scaaf_; }
    std::shared_ptr<const Core::LinAlg::Vector<double>> scaam() override { return scaam_; }
    std::shared_ptr<const Core::LinAlg::Vector<double>> hist() override { return hist_; }
    std::shared_ptr<const Core::LinAlg::Vector<double>> grid_vel() override { return gridv_; }
    std::shared_ptr<const Core::LinAlg::Vector<double>> grid_veln() override { return gridvn_; }
    virtual std::shared_ptr<Core::LinAlg::Vector<double>> write_access_grid_vel() { return gridv_; }
    std::shared_ptr<const Core::LinAlg::Vector<double>> fs_vel() override
    {
      // get fine-scale part of velocity at time n+alpha_F or n+1
      if (Sep_ != nullptr)
      {
        sep_multiply();
      }

      // set fine-scale velocity for parallel nightly tests
      // separation matrix depends on the number of proc here
      if (turbmodel_ == Inpar::FLUID::multifractal_subgrid_scales and
          params_->sublist("MULTIFRACTAL SUBGRID SCALES").get<bool>("SET_FINE_SCALE_VEL"))
        fsvelaf_->put_scalar(0.01);

      return fsvelaf_;
    }

    /// access to Dirichlet maps
    std::shared_ptr<const Core::LinAlg::MapExtractor> get_dbc_map_extractor() override
    {
      return dbcmaps_;
    }

    /// Expand the Dirichlet DOF set
    ///
    /// The method expands the DOF set (map) which contains the DOFs
    /// subjected to Dirichlet boundary conditions. For instance, the method is
    /// called by the staggered FSI in which the velocities on the FSI
    /// interface are prescribed by the other fields.
    void add_dirich_cond(const std::shared_ptr<const Core::LinAlg::Map> maptoadd) override;

    /// Contract the Dirichlet DOF set
    ///
    /// !!! Be careful using this! You might delete dirichlet values set in the input file !!!
    /// !!! So make sure you are only touching the desired dofs.                           !!!
    ///
    /// The method contracts the DOF set (map) which contains the DOFs
    /// subjected to Dirichlet boundary conditions. This method is
    /// called solely by immersed FSI to remove the Dirichlet values from
    /// the previous solution step before a new set is prescribed.
    void remove_dirich_cond(const std::shared_ptr<const Core::LinAlg::Map> maptoremove) override;

    /// Extract the Dirichlet toggle vector based on Dirichlet BC maps
    ///
    /// This method provides backward compatibility only. Formerly, the Dirichlet conditions
    /// were handled with the Dirichlet toggle vector. Now, they are stored and applied
    /// with maps, ie #dbcmaps_. Eventually, this method will be removed.
    virtual std::shared_ptr<const Core::LinAlg::Vector<double>> dirichlet();

    /// Extract the Inverse Dirichlet toggle vector based on Dirichlet BC maps
    ///
    /// This method provides backward compatibility only. Formerly, the Dirichlet conditions
    /// were handled with the Dirichlet toggle vector. Now, they are stored and applied
    /// with maps, ie #dbcmaps_. Eventually, this method will be removed.
    virtual std::shared_ptr<const Core::LinAlg::Vector<double>> inv_dirichlet();

    //! Return locsys manager
    virtual std::shared_ptr<Core::Conditions::LocsysManager> locsys_manager() { return locsysman_; }

    //! Return wss manager
    virtual std::shared_ptr<FLD::Utils::StressManager> stress_manager() { return stressmanager_; }

    //! Return impedance BC
    virtual std::shared_ptr<FLD::Utils::FluidImpedanceWrapper> impedance_bc()
    {
      return impedancebc_;
    }

    //! Evaluate Dirichlet and Neumann boundary conditions
    virtual void set_dirichlet_neumann_bc();

    //! Apply Dirichlet boundary conditions on provided state vectors
    virtual void apply_dirichlet_bc(Teuchos::ParameterList& params,
        std::shared_ptr<Core::LinAlg::Vector<double>> systemvector,    //!< (may be nullptr)
        std::shared_ptr<Core::LinAlg::Vector<double>> systemvectord,   //!< (may be nullptr)
        std::shared_ptr<Core::LinAlg::Vector<double>> systemvectordd,  //!< (may be nullptr)
        bool recreatemap  //!< recreate mapextractor/toggle-vector
                          //!< which stores the DOF IDs subjected
                          //!< to Dirichlet BCs
                          //!< This needs to be true if the bounded DOFs
                          //!< have been changed.
    );

    std::shared_ptr<const Core::LinAlg::Vector<double>> dispnp() override { return dispnp_; }
    virtual std::shared_ptr<Core::LinAlg::Vector<double>> write_access_dispnp() { return dispnp_; }

    //! Create mesh displacement at time level t_{n+1}
    virtual std::shared_ptr<Core::LinAlg::Vector<double>> create_dispnp()
    {
      const Core::LinAlg::Map* aledofrowmap = discret_->dof_row_map(ndsale_);
      dispnp_ = std::make_shared<Core::LinAlg::Vector<double>>(*aledofrowmap, true);
      return dispnp_;
    }

    std::shared_ptr<const Core::LinAlg::Vector<double>> dispn() override { return dispn_; }
    virtual std::shared_ptr<Core::LinAlg::Vector<double>> write_access_dispn() { return dispn_; }

    //! Create mesh displacement at time level t_{n}
    virtual std::shared_ptr<Core::LinAlg::Vector<double>> create_dispn()
    {
      const Core::LinAlg::Map* aledofrowmap = discret_->dof_row_map(ndsale_);
      dispn_ = std::make_shared<Core::LinAlg::Vector<double>>(*aledofrowmap, true);
      return dispn_;
    }
    std::shared_ptr<Core::LinAlg::SparseMatrix> system_matrix() override
    {
      return std::dynamic_pointer_cast<Core::LinAlg::SparseMatrix>(sysmat_);
    }
    std::shared_ptr<Core::LinAlg::SparseMatrix> system_sparse_matrix() override
    {
      return std::dynamic_pointer_cast<Core::LinAlg::BlockSparseMatrixBase>(sysmat_)->merge();
    }
    std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> block_system_matrix() override
    {
      return std::dynamic_pointer_cast<Core::LinAlg::BlockSparseMatrixBase>(sysmat_);
    }
    std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> shape_derivatives() override
    {
      return shapederivatives_;
    }

    virtual std::shared_ptr<Core::LinAlg::MapExtractor> vel_pres_splitter()
    {
      return velpressplitter_;
    };
    std::shared_ptr<const Core::LinAlg::Map> velocity_row_map() override;
    std::shared_ptr<const Core::LinAlg::Map> pressure_row_map() override;
    //  virtual void SetMeshMap(std::shared_ptr<const Core::LinAlg::Map> mm);
    //  double TimeScaling() const;

    /// Use residual_scaling() to convert the implemented fluid residual to an actual force with
    /// unit Newton [N]
    /*! In order to avoid division by time step size \f$\Delta t\f$
     *  the fluid balance of linear momentum is implemented in a way
     *  that the residual does not have the unit Newton [N].
     *  By multiplication with residual_scaling() the residual is
     *  converted to the true residual in unit Newton [N], ie a real force.
     *
     *  \sa trueresidual_
     *  \sa TrueResidual()
     */
    double residual_scaling() const override = 0;

    /*!
    \brief return scheme-specific time integration parameter

    */
    double tim_int_param() const override = 0;

    /*!
    \brief compute values at intermediate time steps for gen.-alpha
           for given vectors and store result in given vectors
           Helper method which can be called from outside fluid (e.g. for coupled problems)

    */
    virtual void gen_alpha_intermediate_values(std::shared_ptr<Core::LinAlg::Vector<double>>& vecnp,
        std::shared_ptr<Core::LinAlg::Vector<double>>& vecn)
    {
      return;
    }

    /// update velocity increment after Newton step
    void update_newton(std::shared_ptr<const Core::LinAlg::Vector<double>> vel) override;

    //  int Itemax() const { return params_->get<int>("max nonlin iter steps"); }
    void set_itemax(int itemax) override { params_->set<int>("max nonlin iter steps", itemax); }


    /*!
    \brief set scalar fields within outer iteration loop

    */
    void set_iter_scalar_fields(std::shared_ptr<const Core::LinAlg::Vector<double>> scalaraf,
        std::shared_ptr<const Core::LinAlg::Vector<double>> scalaram,
        std::shared_ptr<const Core::LinAlg::Vector<double>> scalardtam,
        std::shared_ptr<Core::FE::Discretization> scatradis, int dofset) override;

    /*!
    \brief set scalar fields

    */
    void set_scalar_fields(std::shared_ptr<const Core::LinAlg::Vector<double>> scalarnp,
        const double thermpressnp,
        std::shared_ptr<const Core::LinAlg::Vector<double>> scatraresidual,
        std::shared_ptr<Core::FE::Discretization> scatradis, const int whichscalar = -1) override;

    /*!
    \brief set velocity field obtained by separate computation

    */
    void set_velocity_field(std::shared_ptr<const Core::LinAlg::Vector<double>> setvelnp) override
    {
      velnp_->update(1.0, *setvelnp, 0.0);
      return;
    }

    /// provide access to turbulence statistics manager
    std::shared_ptr<FLD::TurbulenceStatisticManager> turbulence_statistic_manager() override;
    /// provide access to the box filter for dynamic Smagorinsky model
    std::shared_ptr<FLD::DynSmagFilter> dyn_smag_filter() override;
    /// provide access to the box filter for Vreman model
    std::shared_ptr<FLD::Vreman> vreman() override;

    /// introduce surface split extractor object
    /*!
      This method must (and will) be called during setup with a properly
      initialized extractor object if we are on an ale mesh.
     */
    virtual void set_surface_splitter(const Utils::MapExtractor* surfacesplitter)
    {
      surfacesplitter_ = surfacesplitter;
    }

    /// determine grid velocity
    virtual void update_gridv();

    /// prepare AVM3-based scale separation
    virtual void avm3_preparation();

    /// AVM3-based scale separation
    virtual void avm3_separation();

    /// compute flow rate
    virtual void compute_flow_rates() const;

    /// integrate shape functions at nodes marked by condition
    /*!
      Needed for Mortar coupling at the FSI interface
     */
    virtual std::shared_ptr<Core::LinAlg::Vector<double>> integrate_interface_shape(
        std::string condname);

    /// switch fluid field to block matrix
    virtual void use_block_matrix(
        std::shared_ptr<std::set<int>> condelements,  ///< conditioned elements of fluid
        const Core::LinAlg::MultiMapExtractor&
            domainmaps,  ///< domain maps for split of fluid matrix
        const Core::LinAlg::MultiMapExtractor& rangemaps,  ///< range maps for split of fluid matrix
        bool splitmatrix = true                            ///< flag for split of matrices
    );

    /// switch fluid field to block matrix (choose maps for shape derivatives separately)
    virtual void use_block_matrix(
        std::shared_ptr<std::set<int>> condelements,  ///< conditioned elements of fluid
        const Core::LinAlg::MultiMapExtractor&
            domainmaps,  ///< domain maps for split of fluid matrix
        const Core::LinAlg::MultiMapExtractor& rangemaps,  ///< range maps for split of fluid matrix
        std::shared_ptr<std::set<int>> condelements_shape,  ///< conditioned elements
        const Core::LinAlg::MultiMapExtractor&
            domainmaps_shape,  ///< domain maps for split of shape deriv. matrix
        const Core::LinAlg::MultiMapExtractor&
            rangemaps_shape,     ///< domain maps for split of shape deriv. matrix
        bool splitmatrix = true  ///< flag for split of matrices
    );

    /// linear solve with prescribed dirichlet conditions and without history
    /*!
      This is the linear solve as needed for steepest descent FSI.
     */
    virtual void linear_relaxation_solve(std::shared_ptr<Core::LinAlg::Vector<double>> relax);

    //@}

    //! @name methods for turbulence models

    virtual void apply_scale_separation_for_les();

    virtual void outputof_filtered_vel(std::shared_ptr<Core::LinAlg::Vector<double>> outvec,
        std::shared_ptr<Core::LinAlg::Vector<double>> fsoutvec) = 0;

    virtual void print_turbulence_model();
    //@}

    /// set the initial porosity field
    void set_initial_porosity_field(const PoroElast::InitialField,  ///< type of initial field
                                                                    // const int, ///< type of
                                                                    // initial field
        const int startfuncno                                       ///< number of spatial function
        ) override
    {
      FOUR_C_THROW("not implemented in base class");
    }

    virtual void update_iter_incrementally(
        std::shared_ptr<const Core::LinAlg::Vector<double>> vel  //!< input residual velocities
    );

    //! @name methods for fsi
    /// Extrapolation of vectors from mid-point to end-point t_{n+1}
    virtual std::shared_ptr<Core::LinAlg::Vector<double>> extrapolate_end_point(
        std::shared_ptr<Core::LinAlg::Vector<double>> vecn,  ///< vector at time level t_n
        std::shared_ptr<Core::LinAlg::Vector<double>> vecm  ///< vector at time level of equilibrium
    );
    //@}

    /// apply external forces to the fluid
    void apply_external_forces(std::shared_ptr<Core::LinAlg::MultiVector<double>> fext) override;

    /// create field test
    std::shared_ptr<Core::Utils::ResultTest> create_field_test() override;

    std::shared_ptr<const Core::LinAlg::Vector<double>> convective_vel() override;

    /*! \brief Calculate a integrated divergence operator in vector form
     *
     *   The vector valued operator \f$B\f$ is constructed such that
     *   \f$\int_\Omega div (u) \,\mathrm{d}\Omega = B^T u = 0\f$
     */
    virtual std::shared_ptr<Core::LinAlg::Vector<double>> calc_div_op();

    //! @name Biofilm methods
    //@{

    // set fluid displacement vector due to biofilm growth
    void set_fld_gr_disp(std::shared_ptr<Core::LinAlg::Vector<double>> fluid_growth_disp) override;
    //@}

    /*!
    \brief evaluate and update problem-specific boundary conditions

    */
    virtual void do_problem_specific_boundary_conditions() { return; }

    ///< Print stabilization details to screen
    virtual void print_stabilization_details() const;

    ///< Add contribution to external load vector ( add possibly pre-existing external_loads_);
    void add_contribution_to_external_loads(
        const std::shared_ptr<const Core::LinAlg::Vector<double>> contributing_vector) override;

    ///< Update slave dofs for multifield simulations with fluid mesh tying
    void update_source_dof(Core::LinAlg::Vector<double>& f);

    /** \brief Here additional contributions to the system matrix can be set
     *
     * To enforce weak dirichlet conditions as they arise from meshtying for example, such
     * contributions can be set here and will be assembled into the system matrix
     *
     * \param[in] matrix (size fluid_dof x fluid_dof) linear matrix containing entries that need
     * to be assembled into the overall fluid system matrix
     */
    virtual void set_coupling_contributions(
        std::shared_ptr<const Core::LinAlg::SparseOperator> matrix);

    void reset_external_forces();

    std::shared_ptr<const FLD::Meshtying> get_meshtying() { return meshtying_; }

   protected:
    // don't want = operator and cctor
    // FluidImplicitTimeInt operator = (const FluidImplicitTimeInt& old);
    FluidImplicitTimeInt(const FluidImplicitTimeInt& old);

    /*!
    \brief timeloop break criterion
     */
    virtual bool not_finished() { return step_ < stepmax_ and time_ < maxtime_; }

    /*!
    \brief  increment time and step value

    */
    void increment_time_and_step() override
    {
      step_ += 1;
      time_ += dta_;
    }

    /*!
    \brief call elements to calculate system matrix/rhs and assemble

    */
    virtual void assemble_mat_and_rhs();

    /*!
    \brief call elements to calculate system matrix/rhs and assemble, called from
    assemble_mat_and_rhs

    */
    virtual void evaluate_mat_and_rhs(Teuchos::ParameterList& eleparams);

    /*!
    \brief calculate intermediate solution

    */
    void calc_intermediate_solution() override;

    /*!
    \brief apply Dirichlet boundary conditions to system of equations

    */
    virtual void apply_dirichlet_to_system();

    /*!
    \brief apply weak or mixed hybrid Dirichlet boundary conditions to system of equations

    */
    virtual void apply_nonlinear_boundary_conditions();

    /*!
    \brief update acceleration for generalized-alpha time integration

    */
    virtual void gen_alpha_update_acceleration() { return; }

    /*!
    \brief compute values at intermediate time steps for gen.-alpha

    */
    virtual void gen_alpha_intermediate_values() { return; }

    //! Predict velocities which satisfy exactly the Dirichlet BCs
    //! and the linearised system at the previously converged state.
    //!
    //! This is an implicit predictor, i.e. it calls the solver once.
    virtual void predict_tang_vel_consist_acc();

    /*!
    \brief Update of an Ale field based on the fluid state

    */
    virtual void ale_update(std::string condName);

    /*!
    \brief For a given node, obtain local indices of dofs in a vector (like e.g. velnp)

    */
    void get_dofs_vector_local_indicesfor_node(int nodeGid, Core::LinAlg::Vector<double>& vec,
        bool withPressure, std::vector<int>* dofsLocalInd);

    /*!
    \brief add mat and rhs of edge-based stabilization

    */
    virtual void assemble_edge_based_matand_rhs();

    /*!
    \brief Setup meshtying

    */
    virtual void setup_meshtying();

    /*!
    \brief velocity required for evaluation of related quantities required on element level

    */
    virtual std::shared_ptr<const Core::LinAlg::Vector<double>> evaluation_vel() = 0;

    /*!
      \brief add problem dependent vectors

     */
    virtual void add_problem_dependent_vectors() { return; };

    /*!
    \brief Initialize forcing

    */
    virtual void init_forcing();

    /*!
    \brief calculate lift&drag forces and angular momenta

    */
    void lift_drag() const override;

    /**
     * \brief Here, the coupling contributions collected in the linear matrix couplingcontributions_
     * will be added to the system matrix
     */
    virtual void assemble_coupling_contributions();

    //! @name general algorithm parameters

    //! do we move the fluid mesh and calculate the fluid on this moving mesh?
    bool alefluid_;
    //! do we have a turbulence model?
    Inpar::FLUID::TurbModelAction turbmodel_;

    //@}

    /// number of spatial dimensions
    int numdim_;

    //! @name time stepping variables
    int numstasteps_;  ///< number of steps for starting algorithm
    //@}


    /// gas constant (only for low-Mach-number flow)
    double gasconstant_;

    //! use (or not) linearisation of reactive terms on the element
    Inpar::FLUID::LinearisationAction newton_;

    //! kind of predictor used in nonlinear iteration
    std::string predictor_;

    //! @name restart variables
    int writestresses_;
    int write_wall_shear_stresses_;
    int write_eledata_everystep_;
    //@}

    int write_nodedata_first_step_;

    //! @name time step sizes
    double dtp_;  ///< time step size of previous time step
    //@}

    //! @name time-integration-scheme factors
    /// declaration required here in base class
    double theta_;

    //@}

    //! @name parameters for sampling/dumping period
    int samstart_;
    int samstop_;
    int dumperiod_;
    //@}

    std::string statistics_outfilename_;

    //! @name cfl number for adaptive time step
    Inpar::FLUID::AdaptiveTimeStepEstimator cfl_estimator_;  ///< type of adaptive estimator
    double cfl_;                                             ///< cfl number
    //@}

    //! @name norms for convergence check
    double incvelnorm_L2_;
    double incprenorm_L2_;
    double velnorm_L2_;
    double prenorm_L2_;
    double vresnorm_;
    double presnorm_;
    //@}

    //! flag to skip calculation of residual after solution has converged
    bool inconsistent_;

    //! flag to reconstruct second derivative for fluid residual
    bool reconstructder_;

    /// flag for special turbulent flow
    std::string special_flow_;

    /// flag for potential nonlinear boundary conditions
    bool nonlinearbc_;

    /// form of convective term
    std::string convform_;

    /// fine-scale subgrid-viscosity flag
    Inpar::FLUID::FineSubgridVisc fssgv_;

    /// cpu-time measures
    double dtele_;
    double dtfilter_;
    double dtsolve_;

    /// (standard) system matrix
    std::shared_ptr<Core::LinAlg::SparseOperator> sysmat_;

    /// linearization with respect to mesh motion
    std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> shapederivatives_;

    /// maps for extracting Dirichlet and free DOF sets
    std::shared_ptr<Core::LinAlg::MapExtractor> dbcmaps_;

    /// a vector of zeros to be used to enforce zero dirichlet boundary conditions
    std::shared_ptr<Core::LinAlg::Vector<double>> zeros_;

    /// the vector containing body and surface forces
    std::shared_ptr<Core::LinAlg::Vector<double>> neumann_loads_;

    /// the vector containing external loads
    std::shared_ptr<Core::LinAlg::Vector<double>> external_loads_;

    /// the vector containing volume force externally computed
    std::shared_ptr<Core::LinAlg::Vector<double>> forcing_;

    /// the vector containing potential Neumann-type outflow terms
    //  std::shared_ptr<Core::LinAlg::Vector<double>>    outflow_;

    /// a vector containing the integrated traction in boundary normal direction for slip boundary
    /// conditions (Unit: Newton [N])
    std::shared_ptr<Core::LinAlg::Vector<double>> slip_bc_normal_tractions_;

    /// (standard) residual vector (rhs for the incremental form),
    std::shared_ptr<Core::LinAlg::Vector<double>> residual_;

    /// true (rescaled) residual vector without zeros at dirichlet positions (Unit: Newton [N])
    std::shared_ptr<Core::LinAlg::Vector<double>> trueresidual_;

    /// Nonlinear iteration increment vector
    std::shared_ptr<Core::LinAlg::Vector<double>> incvel_;

    //! @name acceleration/(scalar time derivative) at time n+1, n and n+alpha_M/(n+alpha_M/n) and
    //! n-1
    //@{
    std::shared_ptr<Core::LinAlg::Vector<double>> accnp_;  ///< acceleration at time \f$t^{n+1}\f$
    std::shared_ptr<Core::LinAlg::Vector<double>> accn_;   ///< acceleration at time \f$t^{n}\f$
    std::shared_ptr<Core::LinAlg::Vector<double>>
        accam_;  ///< acceleration at time \f$t^{n+\alpha_M}\f$
    std::shared_ptr<Core::LinAlg::Vector<double>> accnm_;  ///< acceleration at time \f$t^{n-1}\f$
    //@}

    //! @name velocity and pressure at time n+1, n, n-1 and n+alpha_F (and n+alpha_M for
    //! weakly_compressible)
    //@{
    std::shared_ptr<Core::LinAlg::Vector<double>> velnp_;  ///< velocity at time \f$t^{n+1}\f$
    std::shared_ptr<Core::LinAlg::Vector<double>> veln_;   ///< velocity at time \f$t^{n}\f$
    std::shared_ptr<Core::LinAlg::Vector<double>>
        velaf_;  ///< velocity at time \f$t^{n+\alpha_F}\f$
    std::shared_ptr<Core::LinAlg::Vector<double>>
        velam_;  ///< velocity at time \f$t^{n+\alpha_M}\f$
    std::shared_ptr<Core::LinAlg::Vector<double>> velnm_;  ///< velocity at time \f$t^{n-1}\f$
    //@}

    //! @name scalar at time n+alpha_F/n+1 and n+alpha_M/n
    std::shared_ptr<Core::LinAlg::Vector<double>> scaaf_;
    std::shared_ptr<Core::LinAlg::Vector<double>> scaam_;
    //@}

    //! @name displacements at time n+1, n and n-1
    //@{
    std::shared_ptr<Core::LinAlg::Vector<double>> dispnp_;  ///< displacement at time \f$t^{n+1}\f$
    std::shared_ptr<Core::LinAlg::Vector<double>> dispn_;   ///< displacement at time \f$t^{n}\f$
    std::shared_ptr<Core::LinAlg::Vector<double>> dispnm_;  ///< displacement at time \f$t^{n-1}\f$
    //@}

    //! @name flow rate and volume at time n+1 (i+1), n+1 (i), n and n-1 for flow-dependent pressure
    //! boundary conditions
    std::vector<double> flowratenp_;
    std::vector<double> flowratenpi_;
    std::vector<double> flowraten_;
    std::vector<double> flowratenm_;

    std::vector<double> flowvolumenp_;
    std::vector<double> flowvolumenpi_;
    std::vector<double> flowvolumen_;
    std::vector<double> flowvolumenm_;

    //@}

    /// only necessary for AVM3: scale-separation matrix
    std::shared_ptr<Core::LinAlg::SparseMatrix> Sep_;

    /// only necessary for AVM3: fine-scale solution vector
    std::shared_ptr<Core::LinAlg::Vector<double>> fsvelaf_;

    /// only necessary for LES models including filtered quantities: filter type
    FLUID::ScaleSeparation scale_sep_;

    /// fine-scale scalar: only necessary for multifractal subgrid-scale modeling in loma
    std::shared_ptr<Core::LinAlg::Vector<double>> fsscaaf_;

    /// grid velocity (set from the adapter!)
    std::shared_ptr<Core::LinAlg::Vector<double>> gridv_;

    /// grid velocity at time step n (set from the adapter!)
    std::shared_ptr<Core::LinAlg::Vector<double>> gridvn_;

    /// histvector --- a linear combination of velnm, veln (BDF)
    ///                or veln, accn (One-Step-Theta)
    std::shared_ptr<Core::LinAlg::Vector<double>> hist_;


    //! manager for turbulence statistics
    std::shared_ptr<FLD::TurbulenceStatisticManager> statisticsmanager_;

    //! forcing for homogeneous isotropic turbulence
    std::shared_ptr<FLD::ForcingInterface> forcing_interface_;

    //! @name Dynamic Smagorinsky model: methods and variables
    //        -------------------------

    //! one instance of the filter object
    std::shared_ptr<FLD::DynSmagFilter> DynSmag_;
    //! one instance of the filter object
    std::shared_ptr<FLD::Vreman> Vrem_;
    std::shared_ptr<FLD::Boxfilter> Boxf_;

    //@}

    //! Extractor to split velnp_ into velocities and pressure DOFs
    //!
    //! velocities  = OtherVector
    //! pressure    = CondVector
    std::shared_ptr<Core::LinAlg::MapExtractor> velpressplitter_;

    /// row dof map extractor
    const Utils::MapExtractor* surfacesplitter_;

    /// a manager doing the transfer of boundary data for
    /// turbulent inflow profiles from a separate (periodic) domain
    std::shared_ptr<TransferTurbulentInflowCondition> turbulent_inflow_condition_;

    /// @name special relaxation state

    bool inrelaxation_;

    std::shared_ptr<Core::LinAlg::SparseMatrix> dirichletlines_;

    std::shared_ptr<Core::LinAlg::SparseMatrix> meshmatrix_;

    /// coupling of fluid-fluid at an internal interface
    std::shared_ptr<FLD::Meshtying> meshtying_;

    /// coupling of fluid-fluid at an internal interface
    std::shared_ptr<FLD::XWall> xwall_;

    /// flag for mesh-tying
    Inpar::FLUID::MeshTying msht_;

    /// face discretization (only initialized for edge-based stabilization)
    std::shared_ptr<Core::FE::DiscretizationFaces> facediscret_;

    //@}

    // possible inf-norm scaling of linear system / fluid matrix
    std::shared_ptr<FLD::Utils::FluidInfNormScaling> fluid_infnormscaling_;

    //! @name Biofilm specific stuff
    //@{
    std::shared_ptr<Core::LinAlg::Vector<double>> fldgrdisp_;
    //@}

    //! Dirichlet BCs with local co-ordinate system
    std::shared_ptr<Core::Conditions::LocsysManager> locsysman_;

    /// windkessel (outflow) boundaries
    std::shared_ptr<Utils::FluidImpedanceWrapper> impedancebc_;

    //! Dirichlet BCs with local co-ordinate system
    std::shared_ptr<FLD::Utils::StressManager> stressmanager_;

    /// flag for windkessel outflow condition
    bool isimpedancebc_;

    /// flag for windkessel outflow condition
    bool off_proc_assembly_;

    /// number of dofset for ALE quantities (0 by default and 2 in case of HDG discretizations)
    unsigned int ndsale_;


    //! @name Set general parameter in class f3Parameter
    /*!

    \brief parameter (fix over a time step) are set in this method.
           Therefore, these parameter are accessible in the fluid element
           and in the fluid boundary element

    */
    virtual void set_element_time_parameter() = 0;

   private:
    //! @name Special method for turbulent variable-density flow at low Mach number with
    //! multifractal subgrid-scale modeling
    /*!

    \brief adaptation of CsgsD to CsgsB
           Since CsgsB depends on the resolution if the near-wall limit is included,
           CsgsD is adapted accordingly by using the mean value of the near-wall  correction.

    */
    virtual void recompute_mean_csgs_b();


    /*! \brief Prepares the locsys manager by calculating the node normals
     *
     */
    void setup_locsys_dirichlet_bc(const double time);

    /// prepares and evaluates edge-based internal face integrals
    void evaluate_fluid_edge_based(std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix1,
        Core::LinAlg::Vector<double>& systemvector1, Teuchos::ParameterList edgebasedparams);

    /*! \brief Compute kinetic energy and write it to file
     *
     *  Kinetic energy of the system is calculated as \f$E_{kin} = \frac{1}{2}u^TMu\f$
     *  with the velocity vector \f$u\f$ and the mass matrix \f$M\f$. Then, it is
     *  written to an output file.
     *
     */
    virtual void write_output_kinetic_energy();

    ///< Evaluate mass matrix
    virtual void evaluate_mass_matrix();

    /// mass matrix (not involved in standard evaluate() since it is invluded in #sysmat_)
    std::shared_ptr<Core::LinAlg::SparseOperator> massmat_;

    /// output stream for energy-file
    std::shared_ptr<std::ofstream> logenergy_;

    /** \brief This matrix can be used to hold contributions to the system matrix like such that
     * arise from meshtying methods or in general weak Dirichlet conditions
     *
     */
    std::shared_ptr<const Core::LinAlg::SparseOperator> couplingcontributions_;
    double meshtyingnorm_;

  };  // class FluidImplicitTimeInt

}  // namespace FLD


FOUR_C_NAMESPACE_CLOSE

#endif
