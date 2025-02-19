// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FLUID_TIMINT_HPP
#define FOUR_C_FLUID_TIMINT_HPP

#include "4C_config.hpp"

#include "4C_adapter_fld_fluid.hpp"
#include "4C_fem_condition.hpp"
#include "4C_fluid_discretization_runtime_output_params.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_utils_exceptions.hpp"

#include <Epetra_Map.h>

#include <memory>

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Core::LinAlg
{
  class Sparsematrix;
  class BlockSparseMatrixBase;
  class MapExtractor;
  class KrylovProjector;
}  // namespace Core::LinAlg

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Core::IO
{
  class DiscretizationWriter;
  class DiscretizationVisualizationWriterMesh;
}  // namespace Core::IO

namespace FLD
{
  class TurbulenceStatisticManager;
  class DynSmagFilter;
  class Vreman;
  namespace Utils
  {
    class KSPMapExtractor;
  }

  class TimInt : public Adapter::Fluid
  {
   public:
    TimInt(const std::shared_ptr<Core::FE::Discretization>& discret,
        const std::shared_ptr<Core::LinAlg::Solver>& solver,
        const std::shared_ptr<Teuchos::ParameterList>& params,
        const std::shared_ptr<Core::IO::DiscretizationWriter>& output);


    void init() override = 0;

    std::shared_ptr<const Core::LinAlg::Vector<double>> initial_guess() override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
      return nullptr;
    }
    std::shared_ptr<const Core::LinAlg::Vector<double>> rhs() override = 0;
    std::shared_ptr<const Core::LinAlg::Vector<double>> true_residual() override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
      return nullptr;
    }
    std::shared_ptr<Core::LinAlg::Vector<double>> write_access_velnp() override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
      return nullptr;
    }
    std::shared_ptr<const Core::LinAlg::Vector<double>> velnp() override = 0;
    std::shared_ptr<const Core::LinAlg::Vector<double>> velaf() override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
      return nullptr;
    }
    std::shared_ptr<const Core::LinAlg::Vector<double>> veln() override = 0;
    std::shared_ptr<const Core::LinAlg::Vector<double>> velnm() override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
      return nullptr;
    }
    std::shared_ptr<const Core::LinAlg::Vector<double>> accnp() override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
      return nullptr;
    }
    std::shared_ptr<const Core::LinAlg::Vector<double>> accn() override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
      return nullptr;
    }
    std::shared_ptr<const Core::LinAlg::Vector<double>> accnm() override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
      return nullptr;
    }
    std::shared_ptr<const Core::LinAlg::Vector<double>> accam() override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
      return nullptr;
    }
    std::shared_ptr<const Core::LinAlg::Vector<double>> scaaf() override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
      return nullptr;
    }
    std::shared_ptr<const Core::LinAlg::Vector<double>> scaam() override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
      return nullptr;
    }
    std::shared_ptr<const Core::LinAlg::Vector<double>> hist() override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
      return nullptr;
    }
    std::shared_ptr<const Core::LinAlg::Vector<double>> grid_vel() override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
      return nullptr;
    }
    std::shared_ptr<const Core::LinAlg::Vector<double>> grid_veln() override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
      return nullptr;
    }
    std::shared_ptr<const Core::LinAlg::Vector<double>> dispnp() override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
      return nullptr;
    }
    std::shared_ptr<const Core::LinAlg::Vector<double>> dispn() override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
      return nullptr;
    }
    std::shared_ptr<const Core::LinAlg::Vector<double>> convective_vel() override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
      return nullptr;
    }
    std::shared_ptr<const Core::LinAlg::Vector<double>> fs_vel() override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
      return nullptr;
    }
    std::shared_ptr<Core::LinAlg::Vector<double>> std_veln() override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
      return nullptr;
    }
    std::shared_ptr<Core::LinAlg::Vector<double>> std_velnp() override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
      return nullptr;
    }
    std::shared_ptr<Core::LinAlg::Vector<double>> std_velaf() override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
      return nullptr;
    }
    std::shared_ptr<const Epetra_Map> dof_row_map() override { return dof_row_map(0); }
    std::shared_ptr<const Epetra_Map> dof_row_map(unsigned nds) override;
    std::shared_ptr<Core::LinAlg::SparseMatrix> system_matrix() override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
      return nullptr;
    }
    std::shared_ptr<Core::LinAlg::SparseMatrix> system_sparse_matrix() override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
      return nullptr;
    }
    std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> block_system_matrix() override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
      return nullptr;
    }
    std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> shape_derivatives() override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
      return nullptr;
    }
    const std::shared_ptr<Core::FE::Discretization>& discretization() override { return discret_; }
    std::shared_ptr<const Core::DOFSets::DofSet> dof_set() override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
      return nullptr;
    }
    std::shared_ptr<const Core::LinAlg::Vector<double>> stepinc() override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
      return nullptr;
    }
    std::shared_ptr<const Core::LinAlg::MapExtractor> get_dbc_map_extractor() override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
      return nullptr;
    }
    void integrate() override = 0;
    void prepare_time_step() override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
    }
    void prepare_solve() override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
    }
    void evaluate(std::shared_ptr<const Core::LinAlg::Vector<double>> vel) override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
    }
    virtual bool convergence_check(int itnum, int itmax, const double ittol)
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
      return false;
    }
    void iter_update(const std::shared_ptr<const Core::LinAlg::Vector<double>> increment) override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
    }
    void update() override = 0;
    void statistics_and_output() override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
    }
    void output() override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
    }
    void statistics_output() override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
    }
    const std::shared_ptr<Core::IO::DiscretizationWriter>& disc_writer() override
    {
      return output_;
    }
    std::shared_ptr<Core::LinAlg::MapExtractor> get_vel_press_splitter() override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
      return nullptr;
    }


    void solve() override = 0;

    void calc_intermediate_solution() override
    {
      FOUR_C_THROW("Not implemented in the base class");
    }
    /// get the linear solver object used for this field
    std::shared_ptr<Core::LinAlg::Solver> linear_solver() override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
      return nullptr;
    }
    std::shared_ptr<const Epetra_Map> inner_velocity_row_map() override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
      return nullptr;
    }  // only used for FSI
    std::shared_ptr<const Epetra_Map> velocity_row_map() override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
      return nullptr;
    }
    std::shared_ptr<const Epetra_Map> pressure_row_map() override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
      return nullptr;
    }

    /// preparations for Krylov space projection
    virtual void setup_krylov_space_projection(Core::Conditions::Condition* kspcond)
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
    }
    virtual void update_krylov_space_projection()
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
    }
    virtual void check_matrix_nullspace()
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
    }


    /// the mesh map contains all velocity dofs that are covered by an ALE node
    void set_mesh_map(std::shared_ptr<const Epetra_Map> mm, const int nds_master = 0) override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
      return;
    }

    /// scaling factor needed to convert the residual to real forces
    double residual_scaling() const override = 0;

    double time_scaling() const override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
      return 1337.0;
    }

    /// return time integration factor
    double tim_int_param() const override = 0;

    /// communication object at the interface (neglecting pressure dofs)
    std::shared_ptr<FLD::Utils::MapExtractor> const& interface() const override
    {
      FOUR_C_THROW("Implemented in the fluid wrapper and derived classes");
      static std::shared_ptr<FLD::Utils::MapExtractor> ret = nullptr;
      return ret;
    }

    /// communication object at the interface needed for fpsi problems (including pressure dofs)
    std::shared_ptr<FLD::Utils::MapExtractor> const& fpsi_interface() const override
    {
      FOUR_C_THROW("Implemented in the fluid wrapper and derived classes");
      static std::shared_ptr<FLD::Utils::MapExtractor> ret = nullptr;
      return ret;
    }

    void read_restart(int step) override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
      return;
    }

    void set_restart(const int step, const double time,
        std::shared_ptr<const Core::LinAlg::Vector<double>> readvelnp,
        std::shared_ptr<const Core::LinAlg::Vector<double>> readveln,
        std::shared_ptr<const Core::LinAlg::Vector<double>> readvelnm,
        std::shared_ptr<const Core::LinAlg::Vector<double>> readaccnp,
        std::shared_ptr<const Core::LinAlg::Vector<double>> readaccn) override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
      return;
    }

    double time() const override { return time_; }
    int step() const override { return step_; }
    double dt() const override { return dta_; }

    //! increment time and step value
    void increment_time_and_step() override;

    //! @name Time step size adaptivity in monolithic FSI
    //@{

    /*! Do one step with auxiliary time integration scheme
     *
     *  Do a single time step with the user given auxiliary time integration
     *  scheme. Result is stored in #locerrvelnp_ and is used later to estimate
     *  the local discretization error of the marching time integration scheme.
     *
     *  \author mayr.mt \date 12/2013
     */
    void time_step_auxiliary() override
    {
      FOUR_C_THROW(
          "We do this in the Adapter until time adaptivity is available in the fluid field.");
    }

    /*! Indicate norms of temporal discretization error
     *
     *  \author mayr.mt \date 12/2013
     */
    void indicate_error_norms(
        double& err,       ///< L2-norm of temporal discretization error based on all DOFs
        double& errcond,   ///< L2-norm of temporal discretization error based on interface DOFs
        double& errother,  ///< L2-norm of temporal discretization error based on interior DOFs
        double& errinf,    ///< L-inf-norm of temporal discretization error based on all DOFs
        double&
            errinfcond,  ///< L-inf-norm of temporal discretization error based on interface DOFs
        double& errinfother  ///< L-inf-norm of temporal discretization error based on interior DOFs
        ) override
    {
      FOUR_C_THROW(
          "We do this in the Adapter until time adaptivity is available in the fluid field.");
    }

    //@}

    //! set time step size
    void set_dt(const double dtnew) override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
      return;
    }

    //! set time and step
    void set_time_step(const double time,  ///< time to set
        const int step                     ///< time step number to set
        ) override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
      return;
    }

    /*!
    \brief Reset time step

    In case of time step size adaptivity, time steps might have to be repeated.
    Therefore, we need to reset the solution back to the initial solution of the
    time step.

    \author mayr.mt
    \date 08/2013
    */
    void reset_step() override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
      return;
    }

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
    void reset_time(const double dtold) override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
      return;
    }

    virtual void lift_drag() const = 0;
    double eval_time() const override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
      return 0.0;
    }
    void redistribute(const std::shared_ptr<Core::LinAlg::Graph> nodegraph) override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
      return;
    }

    std::shared_ptr<Core::LinAlg::Vector<double>> extract_interface_forces() override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
      return nullptr;
    }
    virtual std::shared_ptr<Core::LinAlg::Vector<double>> extract_interface_forces_robin()
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
      return nullptr;
    }
    std::shared_ptr<Core::LinAlg::Vector<double>> extract_interface_velnp() override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
      return nullptr;
    }
    std::shared_ptr<Core::LinAlg::Vector<double>> extract_interface_veln() override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
      return nullptr;
    }
    std::shared_ptr<Core::LinAlg::Vector<double>> extract_free_surface_veln() override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
      return nullptr;
    }
    void apply_interface_velocities(std::shared_ptr<Core::LinAlg::Vector<double>> ivel) override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
      return;
    }
    virtual void apply_interface_robin_value(std::shared_ptr<Core::LinAlg::Vector<double>> ivel,
        std::shared_ptr<Core::LinAlg::Vector<double>> iforce)
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
      return;
    }
    /// Apply initial mesh displacement
    void apply_initial_mesh_displacement(
        std::shared_ptr<const Core::LinAlg::Vector<double>> initfluiddisp) override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
      return;
    }
    void apply_mesh_displacement(
        std::shared_ptr<const Core::LinAlg::Vector<double>> fluiddisp) override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
      return;
    }
    void apply_mesh_displacement_increment(
        std::shared_ptr<const Core::LinAlg::Vector<double>> dispstepinc) override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
      return;
    }
    void apply_mesh_velocity(std::shared_ptr<const Core::LinAlg::Vector<double>> gridvel) override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
      return;
    }

    void displacement_to_velocity(std::shared_ptr<Core::LinAlg::Vector<double>> fcx) override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
      return;
    }
    void velocity_to_displacement(std::shared_ptr<Core::LinAlg::Vector<double>> fcx) override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
      return;
    }

    void free_surf_displacement_to_velocity(
        std::shared_ptr<Core::LinAlg::Vector<double>> fcx) override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
      return;
    }
    void free_surf_velocity_to_displacement(
        std::shared_ptr<Core::LinAlg::Vector<double>> fcx) override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
      return;
    }

    int itemax() const override { return itemax_; }
    void set_itemax(int itemax) override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
      return;
    }

    /*!
    \brief return type of time integration scheme

    */
    Inpar::FLUID::TimeIntegrationScheme tim_int_scheme() const override { return timealgo_; }

    std::shared_ptr<Core::LinAlg::Vector<double>> integrate_interface_shape() override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
      return nullptr;
    }

    void use_block_matrix(bool splitmatrix) override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
      return;
    }

    /// linear fluid solve with just a interface load
    /*!
      The very special solve done in steepest descent relaxation
      calculation (and matrix free Newton Krylov).

      \note Can only be called after a valid fluid solve.
    */
    std::shared_ptr<Core::LinAlg::Vector<double>> relaxation_solve(
        std::shared_ptr<Core::LinAlg::Vector<double>> ivel) override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
      return nullptr;
    }

    std::shared_ptr<Core::Utils::ResultTest> create_field_test() override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
      return nullptr;
    }

    std::shared_ptr<const Core::LinAlg::Vector<double>> extract_velocity_part(
        std::shared_ptr<const Core::LinAlg::Vector<double>> velpres) override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
      return nullptr;
    }

    std::shared_ptr<const Core::LinAlg::Vector<double>> extract_pressure_part(
        std::shared_ptr<const Core::LinAlg::Vector<double>> velpres) override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
      return nullptr;
    }

    /// set initial flow field
    void set_initial_flow_field(
        const Inpar::FLUID::InitialField initfield, const int startfuncno) override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
    }
    /// set initial porosity field
    void set_initial_porosity_field(
        const Inpar::PoroElast::InitialField initfield, const int startfuncno) override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
      return;
    }

    /// apply external forces to the fluid
    void apply_external_forces(std::shared_ptr<Core::LinAlg::MultiVector<double>> fext) override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
      return;
    }

    /// apply external forces to the fluid
    void add_contribution_to_external_loads(
        const std::shared_ptr<const Core::LinAlg::Vector<double>> contributing_vector) override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
      return;
    }

    /// expand dirichlet set
    void add_dirich_cond(const std::shared_ptr<const Epetra_Map> maptoadd) override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
      return;
    };

    /// contract dirichlet set
    void remove_dirich_cond(const std::shared_ptr<const Epetra_Map> maptoremove) override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
      return;
    }

    ///  set scalar fields within outer iteration loop
    void set_iter_scalar_fields(std::shared_ptr<const Core::LinAlg::Vector<double>> scalaraf,
        std::shared_ptr<const Core::LinAlg::Vector<double>> scalaram,
        std::shared_ptr<const Core::LinAlg::Vector<double>> scalardtam,
        std::shared_ptr<Core::FE::Discretization> scatradis, int dofset) override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
    }

    void set_loma_iter_scalar_fields(std::shared_ptr<const Core::LinAlg::Vector<double>> scalaraf,
        std::shared_ptr<const Core::LinAlg::Vector<double>> scalaram,
        std::shared_ptr<const Core::LinAlg::Vector<double>> scalardtam,
        std::shared_ptr<const Core::LinAlg::Vector<double>> fsscalaraf, const double thermpressaf,
        const double thermpressam, const double thermpressdtaf, const double thermpressdtam,
        std::shared_ptr<Core::FE::Discretization> scatradis) override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
    }

    /// set scalar fields
    void set_scalar_fields(std::shared_ptr<const Core::LinAlg::Vector<double>> scalarnp,
        const double thermpressnp,
        std::shared_ptr<const Core::LinAlg::Vector<double>> scatraresidual,
        std::shared_ptr<Core::FE::Discretization> scatradis, const int whichscalar = -1) override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
    }

    /// set velocity field (separate computation)
    void set_velocity_field(std::shared_ptr<const Core::LinAlg::Vector<double>> velnp) override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
      return;
    }

    /// provide access to the turbulence statistic manager
    std::shared_ptr<FLD::TurbulenceStatisticManager> turbulence_statistic_manager() override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
      return nullptr;
    };
    /// provide access to the box filter for dynamic Smagorinsky model
    std::shared_ptr<FLD::DynSmagFilter> dyn_smag_filter() override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
      return nullptr;
    };
    std::shared_ptr<FLD::Vreman> vreman() override { return nullptr; };

    /// update velocity increment after Newton step
    void update_newton(std::shared_ptr<const Core::LinAlg::Vector<double>> vel) override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
      return;
    }

    /// reset data for restart of simulation at beginning
    void reset(bool completeReset = false, int numsteps = 1, int iter = -1) override
    {
      FOUR_C_THROW("reset function not implemented for this fluid adapter");
    };

    // set fluid displacement vector due to biofilm growth
    void set_fld_gr_disp(std::shared_ptr<Core::LinAlg::Vector<double>> fluid_growth_disp) override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
      return;
    }

    void calculate_error() override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
      return;
    }

    Inpar::FLUID::PhysicalType physical_type() const override { return physicaltype_; }

   protected:
    //! fluid discretization
    std::shared_ptr<Core::FE::Discretization> discret_;

    //! linear solver
    std::shared_ptr<Core::LinAlg::Solver> solver_;

    //! parameter list
    std::shared_ptr<Teuchos::ParameterList> params_;

    //! output writer
    std::shared_ptr<Core::IO::DiscretizationWriter> output_;


    /// runtime output writer
    std::shared_ptr<Core::IO::DiscretizationVisualizationWriterMesh> runtime_output_writer_;

    /// runtime output parameter
    Discret::Elements::FluidRuntimeOutputParams runtime_output_params_;

    //! @name Time loop stuff
    //@{

    double time_;  /// physical time
    int step_;     /// timestep number
    double dta_;   /// time step size of current time step

    int stepmax_;     ///< maximal number of timesteps
    double maxtime_;  ///< maximal physical computation time
    int itemax_;      /// maximum number of nonlinear iterations

    //@}

    int uprestart_;  ///< write restart data every uprestart_ steps
    int upres_;      ///< write result every upres_ steps

    Inpar::FLUID::TimeIntegrationScheme timealgo_;  ///< time algorithm flag
    Inpar::FLUID::PhysicalType
        physicaltype_;  ///< flag for physical type of fluid flow (standard: incompressible)

    int myrank_;  ///< the processor ID from the communicator

    //! @name variables for Krylov Space projection
    //@{

    //! flag setting whether Krylov projection needs to be updated or not
    bool updateprojection_;

    //! Krylov projector himself
    std::shared_ptr<Core::LinAlg::KrylovProjector> projector_;

    /// Krylov space projection map extractor
    std::shared_ptr<FLD::Utils::KSPMapExtractor> kspsplitter_;

    //@}

  };  // class TimInt

}  // namespace FLD

FOUR_C_NAMESPACE_CLOSE

#endif
