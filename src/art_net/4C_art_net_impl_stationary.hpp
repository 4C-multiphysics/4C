#ifndef FOUR_C_ART_NET_IMPL_STATIONARY_HPP
#define FOUR_C_ART_NET_IMPL_STATIONARY_HPP

#include "4C_config.hpp"

#include "4C_art_net_timint.hpp"

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
    ArtNetImplStationary(Teuchos::RCP<Core::FE::Discretization> dis, const int linsolvernumber,
        const Teuchos::ParameterList& probparams, const Teuchos::ParameterList& artparams,
        Core::IO::DiscretizationWriter& output);


    // initialization
    void init(const Teuchos::ParameterList& globaltimeparams,
        const Teuchos::ParameterList& arteryparams, const std::string& scatra_disname) override;

    // test results
    void test_results() override;

    // create field test
    Teuchos::RCP<Core::Utils::ResultTest> create_field_test() override;

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

    // output
    void output(bool CoupledTo3D, Teuchos::RCP<Teuchos::ParameterList> CouplingParams) override;

    //! output of element radius
    void output_radius();

    //! output of element volumetric flow
    void output_flow();

    //! set the initial field on the artery discretization
    void set_initial_field(const Inpar::ArtDyn::InitialField init,  //!< type of initial field
        const int startfuncno                                       //!< number of spatial function
        ) override;

    // prepare the loop
    void prepare_time_loop() override;

    // solve artery system of equation
    void solve(Teuchos::RCP<Teuchos::ParameterList> CouplingTo3DParams) override;

    // prepare linear solve (apply DBC)
    void prepare_linear_solve() override;

    // Assembling of the RHS Vector and the LHS Matrix
    void assemble_mat_and_rhs() override;

    // Solve the Linear System of equations
    void linear_solve();

    // Solve Scatra equations
    void solve_scatra() override;

    // get solution vector = pressure
    Teuchos::RCP<const Core::LinAlg::Vector<double>> pressurenp() const override
    {
      return pressurenp_;
    }

    //! get element volume flow
    Teuchos::RCP<const Core::LinAlg::Vector<double>> ele_volflow() const { return ele_volflow_; }

    //! get element radius
    Teuchos::RCP<const Core::LinAlg::Vector<double>> ele_radius() const { return ele_radius_; }

    //! iterative update of primary variable
    void update_iter(const Teuchos::RCP<const Core::LinAlg::Vector<double>> inc) override
    {
      pressurenp_->Update(1.0, *inc, 1.0);
      return;
    }


   private:
    //! a vector of zeros to be used to enforce zero dirichlet boundary conditions
    Teuchos::RCP<Core::LinAlg::Vector<double>> zeros_;
    //! pressure at time n+1
    Teuchos::RCP<Core::LinAlg::Vector<double>> pressurenp_;
    //! pressure increment at time n+1
    Teuchos::RCP<Core::LinAlg::Vector<double>> pressureincnp_;
    //! the vector containing body and surface forces
    Teuchos::RCP<Core::LinAlg::Vector<double>> neumann_loads_;
    //! volumetric flow (for output)
    Teuchos::RCP<Core::LinAlg::Vector<double>> ele_volflow_;
    //! element radius (for output)
    Teuchos::RCP<Core::LinAlg::Vector<double>> ele_radius_;
    /// underlying scatra problem
    Teuchos::RCP<Adapter::ScaTraBaseAlgorithm> scatra_;

  };  // class ArtNetImplStationary
}  // namespace Arteries


FOUR_C_NAMESPACE_CLOSE

#endif
