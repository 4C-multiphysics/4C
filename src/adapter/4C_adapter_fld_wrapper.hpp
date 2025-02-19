// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_ADAPTER_FLD_WRAPPER_HPP
#define FOUR_C_ADAPTER_FLD_WRAPPER_HPP

#include "4C_config.hpp"

#include "4C_adapter_fld_fluid.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Adapter
{
  /// Just wrap, do nothing new, meant to be derived from
  class FluidWrapper : public Fluid
  {
   public:
    explicit FluidWrapper(std::shared_ptr<Fluid> fluid) : fluid_(fluid) {}

    void init() override
    {
      fluid_->init();
      return;
    }
    std::shared_ptr<const Core::LinAlg::Vector<double>> initial_guess() override
    {
      return fluid_->initial_guess();
    }
    std::shared_ptr<const Core::LinAlg::Vector<double>> rhs() override { return fluid_->rhs(); }
    std::shared_ptr<const Core::LinAlg::Vector<double>> true_residual() override
    {
      return fluid_->true_residual();
    }
    std::shared_ptr<const Core::LinAlg::Vector<double>> velnp() override { return fluid_->velnp(); }
    std::shared_ptr<const Core::LinAlg::Vector<double>> velaf() override { return fluid_->velaf(); }
    std::shared_ptr<const Core::LinAlg::Vector<double>> veln() override { return fluid_->veln(); }
    std::shared_ptr<const Core::LinAlg::Vector<double>> velnm() override
    {
      FOUR_C_THROW("not implemented");
      return nullptr;
    };
    std::shared_ptr<const Core::LinAlg::Vector<double>> stepinc() override
    {
      return fluid_->stepinc();
    }
    std::shared_ptr<const Core::LinAlg::Vector<double>> accnp() override
    {
      return fluid_->accnp();
    };
    std::shared_ptr<const Core::LinAlg::Vector<double>> accn() override { return fluid_->accn(); };
    std::shared_ptr<const Core::LinAlg::Vector<double>> accnm() override
    {
      return fluid_->accnm();
    };
    std::shared_ptr<const Core::LinAlg::Vector<double>> accam() override { return fluid_->accam(); }
    std::shared_ptr<const Core::LinAlg::Vector<double>> scaaf() override
    {
      return fluid_->scaaf();
    };
    std::shared_ptr<const Core::LinAlg::Vector<double>> scaam() override
    {
      FOUR_C_THROW("not implemented");
      return nullptr;
    };
    std::shared_ptr<const Core::LinAlg::Vector<double>> hist() override { return fluid_->hist(); }
    std::shared_ptr<const Core::LinAlg::Vector<double>> grid_vel() override
    {
      return fluid_->grid_vel();
    }
    std::shared_ptr<const Core::LinAlg::Vector<double>> grid_veln() override
    {
      return fluid_->grid_veln();
    }
    std::shared_ptr<const Core::LinAlg::Vector<double>> dispnp() override
    {
      return fluid_->dispnp();
    }
    std::shared_ptr<const Core::LinAlg::Vector<double>> dispn() override { return fluid_->dispn(); }
    std::shared_ptr<const Core::LinAlg::Vector<double>> convective_vel() override
    {
      return fluid_->convective_vel();
    }
    std::shared_ptr<const Core::LinAlg::Vector<double>> fs_vel() override
    {
      FOUR_C_THROW("not implemented");
      return nullptr;
    };
    std::shared_ptr<Core::LinAlg::Vector<double>> std_veln() override
    {
      FOUR_C_THROW("not implemented");
      return nullptr;
    };
    std::shared_ptr<Core::LinAlg::Vector<double>> std_velnp() override
    {
      FOUR_C_THROW("not implemented");
      return nullptr;
    };
    std::shared_ptr<Core::LinAlg::Vector<double>> std_velaf() override
    {
      FOUR_C_THROW("not implemented");
      return nullptr;
    };
    std::shared_ptr<const Epetra_Map> dof_row_map() override { return fluid_->dof_row_map(); }
    std::shared_ptr<const Epetra_Map> dof_row_map(unsigned nds) override
    {
      return fluid_->dof_row_map(nds);
    };
    std::shared_ptr<Core::LinAlg::SparseMatrix> system_matrix() override
    {
      return fluid_->system_matrix();
    }
    std::shared_ptr<Core::LinAlg::SparseMatrix> system_sparse_matrix() override
    {
      return fluid_->system_sparse_matrix();
    }
    std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> block_system_matrix() override
    {
      return fluid_->block_system_matrix();
    }
    std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> shape_derivatives() override
    {
      return fluid_->shape_derivatives();
    }
    const std::shared_ptr<Core::FE::Discretization>& discretization() override
    {
      return fluid_->discretization();
    }
    std::shared_ptr<const Core::DOFSets::DofSet> dof_set() override { return fluid_->dof_set(); }
    std::shared_ptr<const Core::LinAlg::MapExtractor> get_dbc_map_extractor() override
    {
      return fluid_->get_dbc_map_extractor();
    }
    void set_initial_flow_field(
        const Inpar::FLUID::InitialField initfield, const int startfuncno) override
    {
      return fluid_->set_initial_flow_field(initfield, startfuncno);
    }
    void set_initial_porosity_field(
        const Inpar::PoroElast::InitialField initfield, const int startfuncno) override
    {
      return fluid_->set_initial_porosity_field(initfield, startfuncno);
    };
    void apply_external_forces(std::shared_ptr<Core::LinAlg::MultiVector<double>> fext) override
    {
      return fluid_->apply_external_forces(fext);
    };
    void add_contribution_to_external_loads(
        const std::shared_ptr<const Core::LinAlg::Vector<double>> contributing_vector) override
    {
      return fluid_->add_contribution_to_external_loads(contributing_vector);
    };
    void add_dirich_cond(const std::shared_ptr<const Epetra_Map> maptoadd) override
    {
      return fluid_->add_dirich_cond(maptoadd);
    };
    void remove_dirich_cond(const std::shared_ptr<const Epetra_Map> maptoremove) override
    {
      return fluid_->remove_dirich_cond(maptoremove);
    };
    void update_newton(std::shared_ptr<const Core::LinAlg::Vector<double>> vel) override
    {
      return fluid_->update_newton(vel);
    };
    void set_loma_iter_scalar_fields(std::shared_ptr<const Core::LinAlg::Vector<double>> scalaraf,
        std::shared_ptr<const Core::LinAlg::Vector<double>> scalaram,
        std::shared_ptr<const Core::LinAlg::Vector<double>> scalardtam,
        std::shared_ptr<const Core::LinAlg::Vector<double>> fsscalaraf, const double thermpressaf,
        const double thermpressam, const double thermpressdtaf, const double thermpressdtam,
        std::shared_ptr<Core::FE::Discretization> scatradis) override
    {
      return fluid_->set_loma_iter_scalar_fields(scalaraf, scalaram, scalardtam, fsscalaraf,
          thermpressaf, thermpressam, thermpressdtaf, thermpressdtam, scatradis);
    }
    void set_iter_scalar_fields(std::shared_ptr<const Core::LinAlg::Vector<double>> scalaraf,
        std::shared_ptr<const Core::LinAlg::Vector<double>> scalaram,
        std::shared_ptr<const Core::LinAlg::Vector<double>> scalardtam,
        std::shared_ptr<Core::FE::Discretization> scatradis, int dofset = 0) override
    {
      return fluid_->set_iter_scalar_fields(scalaraf, scalaram, scalardtam, scatradis, dofset);
    }
    void set_scalar_fields(std::shared_ptr<const Core::LinAlg::Vector<double>> scalarnp,
        const double thermpressnp,
        std::shared_ptr<const Core::LinAlg::Vector<double>> scatraresidual,
        std::shared_ptr<Core::FE::Discretization> scatradis, const int whichscalar = -1) override
    {
      return fluid_->set_scalar_fields(
          scalarnp, thermpressnp, scatraresidual, scatradis, whichscalar);
    }
    std::shared_ptr<FLD::TurbulenceStatisticManager> turbulence_statistic_manager() override
    {
      return fluid_->turbulence_statistic_manager();
    }
    std::shared_ptr<FLD::DynSmagFilter> dyn_smag_filter() override
    {
      return fluid_->dyn_smag_filter();
    }
    std::shared_ptr<FLD::Vreman> vreman() override { return fluid_->vreman(); }
    void set_velocity_field(std::shared_ptr<const Core::LinAlg::Vector<double>> velnp) override
    {
      FOUR_C_THROW("not implemented!");
      return;
    };
    //    virtual void TimeLoop()
    //    { return fluid_->TimeLoop(); }
    void integrate() override { return fluid_->integrate(); }
    void prepare_time_step() override { return fluid_->prepare_time_step(); }
    void increment_time_and_step() override
    {
      FOUR_C_THROW("not implemented!");
      return;
    }
    void prepare_solve() override { fluid_->prepare_solve(); }
    void evaluate(std::shared_ptr<const Core::LinAlg::Vector<double>> stepinc) override
    {
      return fluid_->evaluate(stepinc);
    }
    bool convergence_check(int itnum, int itmax, const double velrestol, const double velinctol,
        const double presrestol, const double presinctol) override
    {
      FOUR_C_THROW("not implemented!");
      return false;
    }
    void iter_update(const std::shared_ptr<const Core::LinAlg::Vector<double>> increment) override
    {
      FOUR_C_THROW("not implemented!");
      return;
    }
    void update() override { return fluid_->update(); }
    void statistics_and_output() override { return fluid_->statistics_and_output(); }
    void output() override { return fluid_->output(); }
    void statistics_output() override
    {
      FOUR_C_THROW("not implemented!");
      return;
    }
    const std::shared_ptr<Core::IO::DiscretizationWriter>& disc_writer() override
    {
      return fluid_->disc_writer();
    }
    std::shared_ptr<Core::LinAlg::MapExtractor> get_vel_press_splitter() override
    {
      return fluid_->get_vel_press_splitter();
    }
    void read_restart(int step) override { return fluid_->read_restart(step); }
    void set_restart(const int step, const double time,
        std::shared_ptr<const Core::LinAlg::Vector<double>> readvelnp,
        std::shared_ptr<const Core::LinAlg::Vector<double>> readveln,
        std::shared_ptr<const Core::LinAlg::Vector<double>> readvelnm,
        std::shared_ptr<const Core::LinAlg::Vector<double>> readaccnp,
        std::shared_ptr<const Core::LinAlg::Vector<double>> readaccn) override
    {
      FOUR_C_THROW("not implemented!");
      return;
    }
    double time() const override { return fluid_->time(); }
    int step() const override { return fluid_->step(); }
    double dt() const override { return fluid_->dt(); }

    //! @name Write access to field solution variables at \f$t^{n+1}\f$
    //@{

    /// write access to extract velocities at \f$t^{n+1}\f$
    std::shared_ptr<Core::LinAlg::Vector<double>> write_access_velnp() override
    {
      return fluid_->write_access_velnp();
    }

    //@}

    //! @name Time step size adaptivity in monolithic FSI
    //@{

    /*! Do one step with auxiliary time integration scheme
     *
     *  Do a single time step with the user given auxiliary time integration
     *  scheme. Result is stored in \p locerrvelnp_ and is used later to estimate
     *  the local discretization error of the marching time integration scheme.
     *
     *  \author mayr.mt \date 12/2013
     */
    void time_step_auxiliary() override {};

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
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
    };

    //! Set fluid time step size that was computed outside
    void set_dt(const double dtnew) override { fluid_->set_dt(dtnew); }

    //! Reset last time step
    void reset_step() override { fluid_->reset_step(); }

    //! Reset time and step number of last time step, needed for time step size adaptivity an FSI
    void reset_time(const double dtold) override { fluid_->reset_time(dtold); }

    //! Set time and step
    void set_time_step(const double time, const int step) override
    {
      fluid_->set_time_step(time, step);
    }

    //@}

    double eval_time() const override
    {
      FOUR_C_THROW("not implemented!");
      return 0.0;
    }
    void redistribute(const std::shared_ptr<Core::LinAlg::Graph> nodegraph) override
    {
      FOUR_C_THROW("not implemented!");
      return;
    }
    void solve() override { return fluid_->solve(); }
    std::shared_ptr<Core::LinAlg::Vector<double>> relaxation_solve(
        std::shared_ptr<Core::LinAlg::Vector<double>> ivel) override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
      return nullptr;
    }
    std::shared_ptr<Core::LinAlg::Solver> linear_solver() override
    {
      return fluid_->linear_solver();
    }
    void calc_intermediate_solution() override { return fluid_->calc_intermediate_solution(); }
    std::shared_ptr<const Epetra_Map> inner_velocity_row_map() override
    {
      return fluid_->inner_velocity_row_map();
    }
    std::shared_ptr<const Epetra_Map> velocity_row_map() override
    {
      return fluid_->velocity_row_map();
    }
    std::shared_ptr<const Epetra_Map> pressure_row_map() override
    {
      return fluid_->pressure_row_map();
    }
    void set_mesh_map(std::shared_ptr<const Epetra_Map> mm, const int nds_master = 0) override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
    }

    /// Use residual_scaling() to convert the implemented fluid residual to an actual force with
    /// unit Newton [N]
    double residual_scaling() const override { return fluid_->residual_scaling(); }

    /// Velocity-displacement conversion at the fsi interface
    double time_scaling() const override { return fluid_->time_scaling(); }

    /// return time integration factor
    double tim_int_param() const override { return fluid_->tim_int_param(); }

    std::shared_ptr<FLD::Utils::MapExtractor> const& interface() const override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
      static std::shared_ptr<FLD::Utils::MapExtractor> ret = nullptr;
      return ret;
    }

    std::shared_ptr<FLD::Utils::MapExtractor> const& fpsi_interface() const override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
      static std::shared_ptr<FLD::Utils::MapExtractor> ret = nullptr;
      return ret;
    }
    Inpar::FLUID::TimeIntegrationScheme tim_int_scheme() const override
    {
      return fluid_->tim_int_scheme();
    }
    std::shared_ptr<const Core::LinAlg::Vector<double>> extract_velocity_part(
        std::shared_ptr<const Core::LinAlg::Vector<double>> velpres) override
    {
      return fluid_->extract_velocity_part(velpres);
    }
    std::shared_ptr<const Core::LinAlg::Vector<double>> extract_pressure_part(
        std::shared_ptr<const Core::LinAlg::Vector<double>> velpres) override
    {
      return fluid_->extract_pressure_part(velpres);
    }
    void apply_interface_velocities(std::shared_ptr<Core::LinAlg::Vector<double>> ivel) override
    {
      return fluid_->apply_interface_velocities(ivel);
    }
    std::shared_ptr<Core::LinAlg::Vector<double>> extract_interface_velnp() override
    {
      return fluid_->extract_interface_velnp();
    }
    std::shared_ptr<Core::LinAlg::Vector<double>> extract_interface_veln() override
    {
      return fluid_->extract_interface_veln();
    }
    std::shared_ptr<Core::LinAlg::Vector<double>> extract_free_surface_veln() override
    {
      return fluid_->extract_free_surface_veln();
    }
    std::shared_ptr<Core::LinAlg::Vector<double>> extract_interface_forces() override
    {
      return fluid_->extract_interface_forces();
    }
    /// Apply initial mesh displacement
    void apply_initial_mesh_displacement(
        std::shared_ptr<const Core::LinAlg::Vector<double>> initfluiddisp) override
    {
      fluid_->apply_initial_mesh_displacement(initfluiddisp);
    }
    void apply_mesh_displacement(
        std::shared_ptr<const Core::LinAlg::Vector<double>> fluiddisp) override
    {
      return fluid_->apply_mesh_displacement(fluiddisp);
    }
    void apply_mesh_displacement_increment(
        std::shared_ptr<const Core::LinAlg::Vector<double>> dispstepinc) override
    {
      return fluid_->apply_mesh_displacement_increment(dispstepinc);
    }
    void apply_mesh_velocity(std::shared_ptr<const Core::LinAlg::Vector<double>> gridvel) override
    {
      return fluid_->apply_mesh_velocity(gridvel);
    }
    void displacement_to_velocity(std::shared_ptr<Core::LinAlg::Vector<double>> fcx) override
    {
      return fluid_->displacement_to_velocity(fcx);
    }
    void velocity_to_displacement(std::shared_ptr<Core::LinAlg::Vector<double>> fcx) override
    {
      return fluid_->velocity_to_displacement(fcx);
    }
    void free_surf_displacement_to_velocity(
        std::shared_ptr<Core::LinAlg::Vector<double>> fcx) override
    {
      return fluid_->free_surf_displacement_to_velocity(fcx);
    }
    void free_surf_velocity_to_displacement(
        std::shared_ptr<Core::LinAlg::Vector<double>> fcx) override
    {
      return fluid_->free_surf_velocity_to_displacement(fcx);
    }
    int itemax() const override { return fluid_->itemax(); }
    void set_itemax(int itemax) override { return fluid_->set_itemax(itemax); }
    std::shared_ptr<Core::LinAlg::Vector<double>> integrate_interface_shape() override
    {
      return fluid_->integrate_interface_shape();
    }
    void use_block_matrix(bool splitmatrix) override
    {
      return fluid_->use_block_matrix(splitmatrix);
    }
    std::shared_ptr<Core::Utils::ResultTest> create_field_test() override
    {
      return fluid_->create_field_test();
    }
    void reset(bool completeReset = false, int numsteps = 1, int iter = -1) override
    {
      return fluid_->reset(completeReset, numsteps, iter);
    };
    void set_fld_gr_disp(std::shared_ptr<Core::LinAlg::Vector<double>> fluid_growth_disp) override
    {
      return fluid_->set_fld_gr_disp(fluid_growth_disp);
    }

    /// calculate error in comparison to analytical solution
    void calculate_error() override
    {
      FOUR_C_THROW("Not implemented in the base class, may be overridden by a subclass.");
    }

    /// return physical type of fluid algorithm
    Inpar::FLUID::PhysicalType physical_type() const override { return fluid_->physical_type(); }

   protected:
    std::shared_ptr<Fluid> fluid_;
  };
}  // namespace Adapter

FOUR_C_NAMESPACE_CLOSE

#endif
