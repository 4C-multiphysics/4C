// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_ADAPTER_FLD_FLUID_HPP
#define FOUR_C_ADAPTER_FLD_FLUID_HPP

#include "4C_config.hpp"

#include "4C_inpar_fluid.hpp"
#include "4C_inpar_poroelast.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_graph.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_parameter_list.fwd.hpp"
#include "4C_utils_result_test.hpp"

#include <Epetra_Map.h>
#include <Epetra_Operator.h>

#include <memory>

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Core::LinAlg
{
  class SparseMatrix;

  class BlockSparseMatrixBase;

  class MapExtractor;

  class Solver;
}  // namespace Core::LinAlg

namespace Core::DOFSets
{
  class DofSet;
}

namespace Core::Elements
{
  class Element;
}

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Core::IO
{
  class DiscretizationWriter;
}

namespace FLD
{
  class TurbulenceStatisticManager;

  class DynSmagFilter;

  class Vreman;
  namespace Utils
  {
    class MapExtractor;
  }
}  // namespace FLD

namespace Adapter
{
  /// general fluid field interface for multiphysics (FSI, ELCH, ...)
  /*!

  This is the FSI algorithm's view on a fluid algorithm. This pure virtual
  interface contains all the methods any FSI algorithm might want to
  call. The idea is to implement this interface with a concrete adapter class
  for each fluid algorithm we want to use for FSI.

  FSI is quite demanding when it comes to knowledge about the internal details
  of a fluid algorithm. Furthermore there are different coupling versions, all
  of them requiring a slightly different view. Yet these views have a lot in
  common, so a common adapter class seems appropriate.

  We use this adapter interface instead of deriving from the fluid algorithm
  class. The good thing is that this way we keep control over the exported
  entities. The down side is that the fluid algorithm still has to grant
  access to a lot of internal state variables.

  Currently supported FSI couplings:

  - Dirichlet-Neumann coupling (fixed-point, Newton-Krylov, vector
    extrapolation)

  - Monolithic (Newton with overlapping blocks and block preconditioning)

  \warning Further cleanup is still needed.

  \sa Structure, Ale
  \author u.kue
  \date 11/07
  */
  class Fluid
  {
   public:
    /// virtual destructor to get polymorph destruction
    virtual ~Fluid() = default;

    /// initialize time integration
    virtual void init() = 0;

    //! @name Vector access

    /// initial guess of Newton's method
    virtual std::shared_ptr<const Core::LinAlg::Vector<double>> initial_guess() = 0;

    /// rhs of Newton's method
    virtual std::shared_ptr<const Core::LinAlg::Vector<double>> rhs() = 0;

    /// true residual
    virtual std::shared_ptr<const Core::LinAlg::Vector<double>> true_residual() = 0;

    /// velocities (and pressures) at \f$t^{n+1}\f$ for write access
    virtual std::shared_ptr<Core::LinAlg::Vector<double>> write_access_velnp() = 0;

    /// velocities (and pressures) at \f$t^{n+1}\f$
    virtual std::shared_ptr<const Core::LinAlg::Vector<double>> velnp() = 0;

    /// velocities (and pressures) at \f$t^{n+\alpha_F}\f$
    virtual std::shared_ptr<const Core::LinAlg::Vector<double>> velaf() = 0;

    /// velocities (and pressures) at \f$t^n\f$
    virtual std::shared_ptr<const Core::LinAlg::Vector<double>> veln() = 0;

    /// velocities (and pressures) at \f$t^{n-1}\f$
    virtual std::shared_ptr<const Core::LinAlg::Vector<double>> velnm() = 0;

    /// accelerations at \f$t^{n+1}\f$
    virtual std::shared_ptr<const Core::LinAlg::Vector<double>> accnp() = 0;

    /// accelerations at \f$t^n\f$
    virtual std::shared_ptr<const Core::LinAlg::Vector<double>> accn() = 0;

    /// accelerations at \f$t^{n-1}\f$
    virtual std::shared_ptr<const Core::LinAlg::Vector<double>> accnm() = 0;

    /// accelerations at \f$t^{n+\alpha_M}\f$
    virtual std::shared_ptr<const Core::LinAlg::Vector<double>> accam() = 0;

    /// scalars at \f$t^{n+\alpha_F}\f$
    virtual std::shared_ptr<const Core::LinAlg::Vector<double>> scaaf() = 0;

    /// scalars at \f$t^{n+\alpha_M}\f$
    virtual std::shared_ptr<const Core::LinAlg::Vector<double>> scaam() = 0;

    /// history vector
    virtual std::shared_ptr<const Core::LinAlg::Vector<double>> hist() = 0;

    /// mesh displacements at \f$t^{n+1}\f$
    virtual std::shared_ptr<const Core::LinAlg::Vector<double>> dispnp() = 0;

    /// mesh displacements at \f$t^{n+1}\f$
    virtual std::shared_ptr<const Core::LinAlg::Vector<double>> dispn() = 0;

    /// convective velocity (= velnp - grid velocity)
    virtual std::shared_ptr<const Core::LinAlg::Vector<double>> convective_vel() = 0;

    /// grid velocity at \f$t^{n+1}\f$
    virtual std::shared_ptr<const Core::LinAlg::Vector<double>> grid_vel() = 0;

    /// grid velocity at \f$t^{n}\f$
    virtual std::shared_ptr<const Core::LinAlg::Vector<double>> grid_veln() = 0;

    /// fine-scale velocity
    virtual std::shared_ptr<const Core::LinAlg::Vector<double>> fs_vel() = 0;

    /// velocities (and pressures) at \f$t^{n}\f$ w/out enriched dofs
    virtual std::shared_ptr<Core::LinAlg::Vector<double>> std_veln() = 0;

    /// velocities (and pressures) at \f$t^{n+1}\f$ w/out enriched dofs
    virtual std::shared_ptr<Core::LinAlg::Vector<double>> std_velnp() = 0;

    /// velocities (and pressures) at \f$t^{n+\alpha_F}\f$ w/out enriched dofs
    virtual std::shared_ptr<Core::LinAlg::Vector<double>> std_velaf() = 0;

    //@}

    //! @name Misc

    /// dof map of vector of unknowns
    virtual std::shared_ptr<const Epetra_Map> dof_row_map() = 0;

    /// dof map of vector of unknowns for multiple dofsets
    virtual std::shared_ptr<const Epetra_Map> dof_row_map(unsigned nds) = 0;

    /// direct access to system matrix
    virtual std::shared_ptr<Core::LinAlg::SparseMatrix> system_matrix() = 0;

    /// direct access to merged system matrix
    virtual std::shared_ptr<Core::LinAlg::SparseMatrix> system_sparse_matrix() = 0;

    /// direct access to system matrix
    virtual std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> block_system_matrix() = 0;

    /// linearization of Navier-Stokes with respect to mesh movement
    virtual std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> shape_derivatives() = 0;

    /// direct access to discretization
    virtual const std::shared_ptr<Core::FE::Discretization>& discretization() = 0;

    /// direct access to dofset
    virtual std::shared_ptr<const Core::DOFSets::DofSet> dof_set() = 0;

    /// Return MapExtractor for Dirichlet boundary conditions
    virtual std::shared_ptr<const Core::LinAlg::MapExtractor> get_dbc_map_extractor() = 0;

    /// set initial flow field
    virtual void set_initial_flow_field(
        const Inpar::FLUID::InitialField initfield, const int startfuncno) = 0;

    /// set initial flow field
    virtual void set_initial_porosity_field(
        const Inpar::PoroElast::InitialField initfield, const int startfuncno) = 0;

    /// apply external forces to the fluid
    virtual void apply_external_forces(std::shared_ptr<Core::LinAlg::MultiVector<double>> fext) = 0;

    /// apply contribution to neumann loads of the fluid (similar to ApplyExternalForces but without
    /// residual scaling)
    virtual void add_contribution_to_external_loads(
        const std::shared_ptr<const Core::LinAlg::Vector<double>> contributing_vector) = 0;

    /// expand dirichlet dbc set by provided map containing dofs to add
    virtual void add_dirich_cond(const std::shared_ptr<const Epetra_Map> maptoadd) = 0;

    /// contract dirichlet set by provided map containing dofs to remove
    virtual void remove_dirich_cond(const std::shared_ptr<const Epetra_Map> maptoremove) = 0;

    ///  set scalar fields within outer iteration loop
    virtual void set_iter_scalar_fields(
        std::shared_ptr<const Core::LinAlg::Vector<double>> scalaraf,
        std::shared_ptr<const Core::LinAlg::Vector<double>> scalaram,
        std::shared_ptr<const Core::LinAlg::Vector<double>> scalardtam,
        std::shared_ptr<Core::FE::Discretization> scatradis, int dofset = 0) = 0;

    /// set scalar fields within outer iteration loop for low-Mach-number flow
    virtual void set_loma_iter_scalar_fields(
        std::shared_ptr<const Core::LinAlg::Vector<double>> scalaraf,
        std::shared_ptr<const Core::LinAlg::Vector<double>> scalaram,
        std::shared_ptr<const Core::LinAlg::Vector<double>> scalardtam,
        std::shared_ptr<const Core::LinAlg::Vector<double>> fsscalaraf, const double thermpressaf,
        const double thermpressam, const double thermpressdtaf, const double thermpressdtam,
        std::shared_ptr<Core::FE::Discretization> scatradis) = 0;

    /// set scalar fields
    virtual void set_scalar_fields(std::shared_ptr<const Core::LinAlg::Vector<double>> scalarnp,
        const double thermpressnp,
        std::shared_ptr<const Core::LinAlg::Vector<double>> scatraresidual,
        std::shared_ptr<Core::FE::Discretization> scatradis, const int whichscalar = -1) = 0;

    /// set velocity field (separate computation)
    virtual void set_velocity_field(std::shared_ptr<const Core::LinAlg::Vector<double>> velnp) = 0;

    /// provide access to the turbulence statistic manager
    virtual std::shared_ptr<FLD::TurbulenceStatisticManager> turbulence_statistic_manager() = 0;

    /// provide access to the box filter class for dynamic Smaorinsky model
    virtual std::shared_ptr<FLD::DynSmagFilter> dyn_smag_filter() = 0;

    virtual std::shared_ptr<FLD::Vreman> vreman() = 0;

    /// reset state vectors (needed for biofilm simulations)
    virtual void reset(bool completeReset = false, int numsteps = 1, int iter = -1) = 0;

    /// set fluid displacement vector due to biofilm growth
    virtual void set_fld_gr_disp(
        std::shared_ptr<Core::LinAlg::Vector<double>> fluid_growth_disp) = 0;
    //@}

    //! @name Time step helpers

    /// run a complete simulation (for fluid stand-alone simulations)
    //    virtual void TimeLoop() = 0;
    virtual void integrate() = 0;

    /// start new time step
    virtual void prepare_time_step() = 0;

    /// increment the time and the step
    virtual void increment_time_and_step() = 0;

    /// preparatives for solve
    virtual void prepare_solve() = 0;

    /// update fluid unknowns and evaluate elements
    ///
    /// there are two increments possible
    /// x^n+1_i+1 = x^n+1_i + iterinc, and
    ///
    /// x^n+1_i+1 = x^n     + stepinc
    ///
    /// with n and i being time and Newton iteration step
    virtual void evaluate(std::shared_ptr<const Core::LinAlg::Vector<double>>
            stepinc  ///< increment between time step n and n+1
        ) = 0;

    /// convergence check
    virtual bool convergence_check(int itnum, int itmax, const double velrestol,
        const double velinctol, const double presrestol, const double presinctol) = 0;

    /// update at end of iteration step
    virtual void iter_update(
        const std::shared_ptr<const Core::LinAlg::Vector<double>> increment) = 0;

    /// update at end of time step
    virtual void update() = 0;

    /// update velocity increment after Newton step
    virtual void update_newton(std::shared_ptr<const Core::LinAlg::Vector<double>> vel) = 0;

    /// lift'n'drag forces, statistics time sample and
    /// output of solution and statistics
    virtual void statistics_and_output() = 0;

    /// output results
    virtual void output() = 0;

    /// output statistics
    virtual void statistics_output() = 0;

    /// access to output
    virtual const std::shared_ptr<Core::IO::DiscretizationWriter>& disc_writer() = 0;

    /// access to map extractor for velocity and pressure
    virtual std::shared_ptr<Core::LinAlg::MapExtractor> get_vel_press_splitter() = 0;

    /// read restart information for given time step
    virtual void read_restart(int step) = 0;

    /// set restart
    virtual void set_restart(const int step, const double time,
        std::shared_ptr<const Core::LinAlg::Vector<double>> readvelnp,
        std::shared_ptr<const Core::LinAlg::Vector<double>> readveln,
        std::shared_ptr<const Core::LinAlg::Vector<double>> readvelnm,
        std::shared_ptr<const Core::LinAlg::Vector<double>> readaccnp,
        std::shared_ptr<const Core::LinAlg::Vector<double>> readaccn) = 0;

    /// current time value
    virtual double time() const = 0;

    /// current time step
    virtual int step() const = 0;

    /// time step size
    virtual double dt() const = 0;

    //! @name Time step size adaptivity in monolithic FSI
    //@{

    /*! Do one step with auxiliary time integration scheme
     *
     *  Do a single time step with the user given auxiliary time integration
     *  scheme. Result is stored in \p locerrvelnp_ and is used later to
     *  estimate the local discretization error of the marching time integration
     *  scheme.
     *
     *  \author mayr.mt \date 12/2013
     */
    virtual void time_step_auxiliary() = 0;

    /*! Indicate norms of local discretization error
     *
     *  \author mayr.mt \date 12/2013
     */
    virtual void indicate_error_norms(
        double& err,       ///< L2-norm of temporal discretization error based on all DOFs
        double& errcond,   ///< L2-norm of temporal discretization error based on interface DOFs
        double& errother,  ///< L2-norm of temporal discretization error based on interior DOFs
        double& errinf,    ///< L-inf-norm of temporal discretization error based on all DOFs
        double&
            errinfcond,  ///< L-inf-norm of temporal discretization error based on interface DOFs
        double& errinfother  ///< L-inf-norm of temporal discretization error based on interior DOFs
        ) = 0;

    /// set time step size
    virtual void set_dt(const double dtold) = 0;

    /// set time and step
    virtual void set_time_step(const double time,  ///< time to set
        const int step                             ///< time step number to set
        ) = 0;

    //@}

    /*!
    \brief Reset time step

    In case of time step size adaptivity, time steps might have to be repeated.
    Therefore, we need to reset the solution back to the initial solution of the
    time step.

    \author mayr.mt
    \date 08/2013
    */
    virtual void reset_step() = 0;

    /*!
    \brief Reset time and step in case that a time step has to be repeated

    Fluid field increments time and step at the beginning of a time step. If a time
    step has to be repeated, we need to take this into account and decrease time and
    step beforehand. They will be incremented right at the beginning of the repetition
    and, thus, everything will be fine. Currently, this is needed for time step size
    adaptivity in FSI.

    \author mayr.mt
    \date 08/2013
     */
    virtual void reset_time(const double dtold) = 0;

    /// this procs element evaluate time
    virtual double eval_time() const = 0;

    /// redistribute the fluid discretization and vectors according to nodegraph in std. mode
    virtual void redistribute(const std::shared_ptr<Core::LinAlg::Graph> nodegraph) = 0;


    //@}

    //! @name Solver calls

    /// nonlinear solve
    /*!
      Do the nonlinear solve for the time step. All boundary conditions have
      been set.
     */
    virtual void solve() = 0;

    /// linear fluid solve with just a interface load
    virtual std::shared_ptr<Core::LinAlg::Vector<double>> relaxation_solve(
        std::shared_ptr<Core::LinAlg::Vector<double>> ivel) = 0;

    /// get the linear solver object used for this field
    virtual std::shared_ptr<Core::LinAlg::Solver> linear_solver() = 0;

    /// do an intermediate solution step
    virtual void calc_intermediate_solution() = 0;

    //@}

    /// Map of all velocity dofs that are not Dirichlet-constrained
    virtual std::shared_ptr<const Epetra_Map> inner_velocity_row_map() = 0;

    /// Map of all velocity dofs
    virtual std::shared_ptr<const Epetra_Map> velocity_row_map() = 0;

    /// Map of all pressure dofs
    virtual std::shared_ptr<const Epetra_Map> pressure_row_map() = 0;

    /// the mesh map contains all velocity dofs that are covered by an ALE node
    virtual void set_mesh_map(std::shared_ptr<const Epetra_Map> mm, const int nds_master = 0) = 0;

    /// Use residual_scaling() to convert the implemented fluid residual to an actual force with
    /// unit Newton [N]
    virtual double residual_scaling() const = 0;

    /// Velocity-displacement conversion at the fsi interface
    /*! Time integration of the fsi interface reads:
     *  \f$\mathbf{d}^{n+1} = \mathbf{d}^{n} + \tau(\mathbf{u}^{n+1}-\mathbf{u}^{n}) + \Delta t
     * \mathbf{u}^{n}\f$
     *
     *  Currently, two time integration schemes for the fsi interface
     *  are implemented:
     *  - Backward-Euler: \f$\tau = \Delta t\f$
     *  - Trapezoidal rule: \f$\tau = \frac{\Delta t}{2}\f$
     *
     *  Use TimeScaling() to get \f$\tau=\frac{1}{\text{TimeScaling()}}\f$
     */
    virtual double time_scaling() const = 0;

    /// return time integration factor
    virtual double tim_int_param() const = 0;

    /// communication object at the interface (neglecting pressure dofs)
    virtual std::shared_ptr<FLD::Utils::MapExtractor> const& interface() const = 0;

    /// communication object at the interface (including pressure dofs)
    virtual std::shared_ptr<FLD::Utils::MapExtractor> const& fpsi_interface() const = 0;

    /// return type of time integration scheme
    virtual Inpar::FLUID::TimeIntegrationScheme tim_int_scheme() const = 0;

    //! @name Extract the velocity-related part of a fluid vector (e.g. velnp, veln, residual)
    /// The idea is to have one function that does the extraction and call it
    /// with different vectors.

    /// Some applications need only access to velocity-related values of an fluid result vector.
    virtual std::shared_ptr<const Core::LinAlg::Vector<double>> extract_velocity_part(
        std::shared_ptr<const Core::LinAlg::Vector<double>> velpres) = 0;

    /// Some applications need only access to pressure-related values of an fluid result vector.
    virtual std::shared_ptr<const Core::LinAlg::Vector<double>> extract_pressure_part(
        std::shared_ptr<const Core::LinAlg::Vector<double>> velpres) = 0;

    //@}

    //! @name Apply interface values

    /// at the interface the velocity is prescribed as a Dirichlet condition
    virtual void apply_interface_velocities(std::shared_ptr<Core::LinAlg::Vector<double>> ivel) = 0;

    //@}

    //! @name Extract interface values
    /// Maybe we do not need all of them?

    /// extract fluid velocity at the interface from time step n+1
    virtual std::shared_ptr<Core::LinAlg::Vector<double>> extract_interface_velnp() = 0;

    /// extract fluid velocity at the interface from time step n
    virtual std::shared_ptr<Core::LinAlg::Vector<double>> extract_interface_veln() = 0;

    /// extract fluid velocity at the free surface from time step n
    virtual std::shared_ptr<Core::LinAlg::Vector<double>> extract_free_surface_veln() = 0;

    /// extract fluid forces at the interface
    virtual std::shared_ptr<Core::LinAlg::Vector<double>> extract_interface_forces() = 0;

    virtual std::shared_ptr<const Core::LinAlg::Vector<double>> stepinc() = 0;

    //@}

    //! @name Extract mesh values

    /// tell the initial mesh displacement to the fluid solver
    virtual void apply_initial_mesh_displacement(
        std::shared_ptr<const Core::LinAlg::Vector<double>> initfluiddisp) = 0;

    /// tell the mesh displacement to the fluid solver
    virtual void apply_mesh_displacement(
        std::shared_ptr<const Core::LinAlg::Vector<double>> fluiddisp) = 0;

    /// tell the mesh displacement step increment to the fluid solver
    virtual void apply_mesh_displacement_increment(
        std::shared_ptr<const Core::LinAlg::Vector<double>> dispstepinc) = 0;

    /// tell the mesh velocity to the fluid solver
    virtual void apply_mesh_velocity(
        std::shared_ptr<const Core::LinAlg::Vector<double>> gridvel) = 0;

    //@}

    //! @name Conversion between displacement and velocity at interface

    /// convert Delta d(n+1,i+1) to the fluid unknown at the interface
    virtual void displacement_to_velocity(std::shared_ptr<Core::LinAlg::Vector<double>> fcx) = 0;

    /// convert the fluid unknown to Delta d(n+1,i+1) at the interface
    virtual void velocity_to_displacement(std::shared_ptr<Core::LinAlg::Vector<double>> fcx) = 0;

    /// convert Delta d(n+1,i+1) to the fluid unknown at the free surface
    virtual void free_surf_displacement_to_velocity(
        std::shared_ptr<Core::LinAlg::Vector<double>> fcx) = 0;

    /// convert the fluid unknown to Delta d(n+1,i+1) at the free surface
    virtual void free_surf_velocity_to_displacement(
        std::shared_ptr<Core::LinAlg::Vector<double>> fcx) = 0;

    //@}

    //! @name Number of Newton iterations
    //! For simplified FD MFNK solve we want to temporally limit the
    /// number of Newton steps inside the fluid solver

    /// return maximum for iteration steps
    virtual int itemax() const = 0;

    /// set maximum for iteration steps
    virtual void set_itemax(int itemax) = 0;

    //@}

    /// integrate FSI interface shape functions
    virtual std::shared_ptr<Core::LinAlg::Vector<double>> integrate_interface_shape() = 0;

    /// switch fluid field to block matrix
    virtual void use_block_matrix(bool splitmatrix) = 0;

    /// create result test for encapulated fluid algorithm
    virtual std::shared_ptr<Core::Utils::ResultTest> create_field_test() = 0;

    /// calculate error in comparison to analytical solution
    virtual void calculate_error() = 0;

    /// return physical type of fluid algorithm
    virtual Inpar::FLUID::PhysicalType physical_type() const = 0;
  };
}  // namespace Adapter

FOUR_C_NAMESPACE_CLOSE

#endif
