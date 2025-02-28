// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_ART_NET_IMPL_STATIONARY_HPP
#define FOUR_C_ART_NET_IMPL_STATIONARY_HPP

#include "4C_config.hpp"

#include "4C_art_net_timint.hpp"
#include "4C_io_discretization_visualization_writer_mesh.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Adapter
{
  class ScaTraBaseAlgorithm;
}

namespace Arteries
{
  /*!
  \brief stationary formulation for arterial network problems

  \author kremheller
  */
  class ArtNetImplStationary : public TimInt
  {
   public:
    /*!
    \brief Standard Constructor

    */
    ArtNetImplStationary(std::shared_ptr<Core::FE::Discretization> dis, const int linsolvernumber,
        const Teuchos::ParameterList& probparams, const Teuchos::ParameterList& artparams,
        Core::IO::DiscretizationWriter& output);


    // initialization
    void init(const Teuchos::ParameterList& globaltimeparams,
        const Teuchos::ParameterList& arteryparams, const std::string& scatra_disname) override;

    // test results
    void test_results() override;

    // create field test
    std::shared_ptr<Core::Utils::ResultTest> create_field_test() override;

    /// setup the variables to do a new time step
    void time_update() override;

    /// prepare time step
    void prepare_time_step() override;

    /// setup Dirichlet Boundary conditions
    void apply_dirichlet_bc();

    /// reset artery diameter of previous time step
    void reset_artery_diam_previous_time_step();

    //! Apply Neumann boundary conditions
    void apply_neumann_bc(Core::LinAlg::Vector<double>& neumann_loads  //!< Neumann loads
    );

    /// add neumann BC to residual
    void add_neumann_to_residual();

    /// initialization
    void init_save_state() override
    {
      FOUR_C_THROW("InitSaveState() not available for stationary formulation");
    }

    // restart
    void read_restart(int step, bool CoupledTo3D = false) override;

    /// save state
    void save_state() override
    {
      FOUR_C_THROW("SaveState() not available for stationary formulation");
    }

    void load_state() override
    {
      FOUR_C_THROW("LoadState() not available for stationary formulation");
    }

    //! collect runtime output data
    void collect_runtime_output_data();

    //! write data required for restart
    virtual void output_restart();

    // output
    void output(bool CoupledTo3D, std::shared_ptr<Teuchos::ParameterList> CouplingParams) override;

    //! get element radius
    void get_radius();

    //! calculate element volumetric flow
    void reconstruct_flow();

    //! set the initial field on the artery discretization
    void set_initial_field(const Inpar::ArtDyn::InitialField init,  //!< type of initial field
        const int startfuncno                                       //!< number of spatial function
        ) override;

    // prepare the loop
    void prepare_time_loop() override;

    // solve artery system of equation
    void solve(std::shared_ptr<Teuchos::ParameterList> CouplingTo3DParams) override;

    // prepare linear solve (apply DBC)
    void prepare_linear_solve() override;

    // Assembling of the RHS Vector and the LHS Matrix
    void assemble_mat_and_rhs() override;

    // Solve the Linear System of equations
    void linear_solve();

    // Solve Scatra equations
    void solve_scatra() override;

    // get solution vector = pressure
    std::shared_ptr<const Core::LinAlg::Vector<double>> pressurenp() const override
    {
      return pressurenp_;
    }

    //! get element volume flow
    std::shared_ptr<const Core::LinAlg::Vector<double>> ele_volflow() const { return ele_volflow_; }

    //! get element radius
    std::shared_ptr<const Core::LinAlg::Vector<double>> ele_radius() const { return ele_radius_; }

    //! iterative update of primary variable
    void update_iter(const std::shared_ptr<const Core::LinAlg::Vector<double>> inc) override
    {
      pressurenp_->Update(1.0, *inc, 1.0);
      return;
    }


   private:
    std::unique_ptr<Core::IO::DiscretizationVisualizationWriterMesh> visualization_writer_{nullptr};
    //! a vector of zeros to be used to enforce zero dirichlet boundary conditions
    std::shared_ptr<Core::LinAlg::Vector<double>> zeros_;
    //! pressure at time n+1
    std::shared_ptr<Core::LinAlg::Vector<double>> pressurenp_;
    //! pressure increment at time n+1
    std::shared_ptr<Core::LinAlg::Vector<double>> pressureincnp_;
    //! the vector containing body and surface forces
    std::shared_ptr<Core::LinAlg::Vector<double>> neumann_loads_;
    //! volumetric flow (for output)
    std::shared_ptr<Core::LinAlg::Vector<double>> ele_volflow_;
    //! element radius (for output)
    std::shared_ptr<Core::LinAlg::Vector<double>> ele_radius_;
    /// underlying scatra problem
    std::shared_ptr<Adapter::ScaTraBaseAlgorithm> scatra_;

  };  // class ArtNetImplStationary
}  // namespace Arteries


FOUR_C_NAMESPACE_CLOSE

#endif
