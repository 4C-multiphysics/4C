// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_LEVELSET_TIMINT_OST_HPP
#define FOUR_C_LEVELSET_TIMINT_OST_HPP

#include "4C_config.hpp"

#include "4C_levelset_algorithm.hpp"
#include "4C_scatra_timint_ost.hpp"

FOUR_C_NAMESPACE_OPEN


namespace ScaTra
{
  class LevelSetTimIntOneStepTheta : public LevelSetAlgorithm, public TimIntOneStepTheta
  {
   public:
    /// standard Constructor
    LevelSetTimIntOneStepTheta(std::shared_ptr<Core::FE::Discretization> dis,
        std::shared_ptr<Core::LinAlg::Solver> solver,
        std::shared_ptr<Teuchos::ParameterList> params,
        std::shared_ptr<Teuchos::ParameterList> sctratimintparams,
        std::shared_ptr<Teuchos::ParameterList> extraparams,
        std::shared_ptr<Core::IO::DiscretizationWriter> output);


    /// initialize time-integration scheme
    void init() override;

    /// setup time-integration scheme
    void setup() override;

    /// read restart data
    void read_restart(
        const int step, std::shared_ptr<Core::IO::InputControl> input = nullptr) override;

    /// redistribute the scatra discretization and vectors according to nodegraph
    void redistribute(Core::LinAlg::Graph& nodegraph);

    /// interpolate phi to intermediate time n+theta with 0<theta<1
    std::shared_ptr<Core::LinAlg::Vector<double>> phinptheta(const double theta_inter);

    /// interpolate phidt to intermediate time n+theta with 0<theta<1
    std::shared_ptr<Core::LinAlg::Vector<double>> phidtnptheta(const double theta_inter);

   protected:
    /// Print information about current time step to screen (reimplementation for OST)
    void print_time_step_info() override;

    /// calculate consistent initial scalar time derivatives in compliance with initial scalar field
    void calc_initial_time_derivative() override;

    /// additional predictor not intended for level-set methods
    void explicit_predictor() const override { return; };

    /// Set the part of the righthandside belonging to the last timestep.
    void set_old_part_of_righthandside() override;

    /// update state vectors
    /// current solution becomes old solution of next time step
    void update_state() override;

    /// update the solution after Solve()
    /// extended version for coupled level-set problems including reinitialization
    void update() override;

    /// update phi within the reinitialization loop
    void update_reinit() override;

   private:
  };  // class LevelSetTimIntOneStepTheta

}  // namespace ScaTra

FOUR_C_NAMESPACE_CLOSE

#endif
