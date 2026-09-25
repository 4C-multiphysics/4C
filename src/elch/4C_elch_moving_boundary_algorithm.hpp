// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_ELCH_MOVING_BOUNDARY_ALGORITHM_HPP
#define FOUR_C_ELCH_MOVING_BOUNDARY_ALGORITHM_HPP

#include "4C_config.hpp"

#include "4C_adapter_scatra_fluid_ale_coupling_algo.hpp"

FOUR_C_NAMESPACE_OPEN

namespace ElCh
{
  /// ELCH algorithm with support for deforming meshes
  /*!

    ELCH algorithm with moving meshes. Derives from ScaTraFluidAleCouplingAlgorithm.

   */
  class MovingBoundaryAlgorithm : public Adapter::ScaTraFluidAleCouplingAlgorithm
  {
   public:
    /// constructor
    MovingBoundaryAlgorithm(Global::Problem& problem,  ///< global problem
        MPI_Comm comm,                                 ///< communicator
        const Teuchos::ParameterList& elchcontrol,     ///< elch parameter list
        const Teuchos::ParameterList& scatradyn,       ///< scatra parameter list
        const Teuchos::ParameterList& solverparams     ///< solver parameter list
    );


    /// setup
    void setup() override;

    /// init
    void init() override;

    /// outer level ELCH time loop
    void time_loop() override;

    /// read restart data
    void read_restart(int step) override;

    /// Add tests to global problem and start tests
    void test_results();

   protected:
    /// start a new time step
    void prepare_time_step() override;

    /// solve Navier-Stokes and ALE for current time step
    void solve_fluid_ale();

    /// solve transport equations for current time step
    void solve_scatra();

    /// compute interface displacement and velocity
    void compute_interface_vectors(
        Core::LinAlg::Vector<double>& idispnp_, Core::LinAlg::Vector<double>& iveln_);

    /// take current results for converged and save for next time step
    void update() override;

    /// write output
    void output() override;

   private:
    bool pseudotransient_;

    /// molar volume for flux to shape change conversion (unit: m^3/mol )
    const double molarvolume_;

    /// interface displacement at time t^{n}
    std::shared_ptr<Core::LinAlg::Vector<double>> idispn_;

    /// interface displacement at time t^{n+1}
    std::shared_ptr<Core::LinAlg::Vector<double>> idispnp_;

    /// fluid velocity at interface (always zero!)
    std::shared_ptr<Core::LinAlg::Vector<double>> iveln_;

    /// old flux
    std::shared_ptr<Core::LinAlg::MultiVector<double>> fluxn_;

    /// current flux
    std::shared_ptr<Core::LinAlg::MultiVector<double>> fluxnp_;

    /// maximum iteration steps for outer loop
    const int itmax_;

    /// absolute displacement tolerance
    const double ittol_;

    /// parameter for velocity <-> displacement conversion in a OST sense
    const double theta_;

    const Teuchos::ParameterList& elch_params_;
  };

}  // namespace ElCh

FOUR_C_NAMESPACE_CLOSE

#endif
