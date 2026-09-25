// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SCATRA_ALGORITHM_HPP
#define FOUR_C_SCATRA_ALGORITHM_HPP

#include "4C_config.hpp"

#include "4C_adapter_scatra_fluid_coupling_algorithm.hpp"

FOUR_C_NAMESPACE_OPEN

namespace ScaTra
{
  /// Basic algorithm for the transport of passive scalars in the fluid velocity field
  /*!

   */
  class ScaTraAlgorithm : public Adapter::ScaTraFluidCouplingAlgorithm
  {
   public:
    /// constructor
    ScaTraAlgorithm(Global::Problem& problem,       ///< global problem
        MPI_Comm comm,                              ///< communicator
        const Teuchos::ParameterList& scatradyn,    ///< scatra parameter list
        const Teuchos::ParameterList& fdyn,         ///< fluid parameter list
        const std::string scatra_disname,           ///< scatra discretization name
        const Teuchos::ParameterList& solverparams  ///< solver parameter list
    );


    void setup() override;

    void init() override;

    /// outer level time loop
    void time_loop() override;

    /// read restart for preceding turbulent inflow simulation
    void read_inflow_restart(int restart);

    /// Add tests to global problem and start tests
    void test_results();

   protected:
    /// time loop for one-way coupling
    void time_loop_one_way();

    /// time loop for two-way coupling (natural convection)
    void time_loop_two_way();

    /// provide information about initial field
    virtual void prepare_time_loop() {};

    /// initial calculations for two-way coupling time loop
    void prepare_time_loop_two_way();

    /// start a new time step
    void prepare_time_step() override;

    /// start a new time step
    void prepare_time_step_convection();

    /// print scatra solver type to screen
    virtual void print_scatra_solver();

    /// solve Navier-Stokes equations for current time step
    void do_fluid_step();

    /// solve transport (convection-diffusion) equations for current time step
    void do_transport_step();

    /// set fluid velocites in scatra field
    void set_velocity_field();

    /// Outer iteration loop for natural convection
    void outer_iteration_convection();

    /// take current results for converged and save for next time step
    void update() override;

    /// take current results for converged and save for next time step
    void update_convection();

    /// write output
    void output() override;

    /// convergence check for natural convection solver
    virtual bool convergence_check(int natconvitnum, int natconvitmax, double natconvittol);

    /// flag for natural convection effects
    int natconv_;

    /// maximum iteration steps of outer loop for natural convection
    const int natconvitmax_;

    /// convection tolerance for outer loop for natural convection
    const double natconvittol_;

    /// outer loop velocity increment for natural convection
    std::shared_ptr<Core::LinAlg::Vector<double>> velincnp_;

    /// outer loop phi increment for natural convection
    std::shared_ptr<Core::LinAlg::Vector<double>> phiincnp_;

    /// start step for sampling of statistical data
    const int samstart_;

    /// end step for sampling of statistical data
    const int samstop_;
  };

}  // namespace ScaTra

FOUR_C_NAMESPACE_CLOSE

#endif
