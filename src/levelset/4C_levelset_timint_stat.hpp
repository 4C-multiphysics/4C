// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_LEVELSET_TIMINT_STAT_HPP
#define FOUR_C_LEVELSET_TIMINT_STAT_HPP

#include "4C_config.hpp"

#include "4C_levelset_algorithm.hpp"
#include "4C_scatra_timint_stat.hpp"

FOUR_C_NAMESPACE_OPEN


namespace ScaTra
{
  class LevelSetTimIntStationary : public LevelSetAlgorithm, public TimIntStationary
  {
   public:
    /// Standard Constructor
    LevelSetTimIntStationary(std::shared_ptr<Core::FE::Discretization> dis,
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
        const int step, std::shared_ptr<Core::IO::InputControl> input = nullptr) override
    {
      FOUR_C_THROW("You should not need this function!");
      return;
    };

    /// redistribute the scatra discretization and vectors according to nodegraph
    void redistribute(Core::LinAlg::Graph& nodegraph)
    {
      FOUR_C_THROW("You should not need this function!");
      return;
    };

   protected:
    /// update state vectors
    /// current solution becomes old solution of next time step
    void update_state() override
    {
      FOUR_C_THROW("You should not need this function!");
      return;
    };

    /// update the solution after Solve()
    /// extended version for coupled level-set problems including reinitialization
    void update() override
    {
      FOUR_C_THROW("You should not need this function!");
      return;
    };

    /// update phi within the reinitialization loop
    void update_reinit() override
    {
      FOUR_C_THROW("You should not need this function!");
      return;
    };

   private:
  };  // class TimIntStationary

}  // namespace ScaTra

FOUR_C_NAMESPACE_CLOSE

#endif
