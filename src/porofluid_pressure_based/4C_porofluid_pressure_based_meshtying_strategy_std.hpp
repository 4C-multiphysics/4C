// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_POROFLUID_PRESSURE_BASED_MESHTYING_STRATEGY_STD_HPP
#define FOUR_C_POROFLUID_PRESSURE_BASED_MESHTYING_STRATEGY_STD_HPP

#include "4C_config.hpp"

#include "4C_porofluid_pressure_based_meshtying_strategy_base.hpp"

FOUR_C_NAMESPACE_OPEN

namespace PoroPressureBased
{
  class MeshtyingStrategyStd : public MeshtyingStrategyBase
  {
   public:
    //! constructor
    explicit MeshtyingStrategyStd(PoroPressureBased::TimIntImpl* porofluidmultitimint,
        const Teuchos::ParameterList& probparams, const Teuchos::ParameterList& poroparams);


    //! prepare time loop
    void prepare_time_loop() override;

    //! prepare time step
    void prepare_time_step() override;

    //! update
    void update() override;

    //! output
    void output() override;

    //! Initialize the linear solver
    void initialize_linear_solver(std::shared_ptr<Core::LinAlg::Solver> solver) override;

    //! solve linear system of equations
    void linear_solve(std::shared_ptr<Core::LinAlg::Solver> solver,
        std::shared_ptr<Core::LinAlg::SparseOperator> sysmat,
        std::shared_ptr<Core::LinAlg::Vector<double>> increment,
        std::shared_ptr<Core::LinAlg::Vector<double>> residual,
        Core::LinAlg::SolverParams& solver_params) override;

    //! calculate norms for convergence checks
    void calculate_norms(std::vector<double>& preresnorm, std::vector<double>& incprenorm,
        std::vector<double>& prenorm,
        const std::shared_ptr<const Core::LinAlg::Vector<double>> increment) override;

    //! create the field test
    void create_field_test() override;

    //! restart
    void read_restart(const int step) override;

    //! evaluate mesh tying
    void evaluate() override;

    //! extract increments and update mesh tying
    std::shared_ptr<const Core::LinAlg::Vector<double>> extract_and_update_iter(
        const std::shared_ptr<const Core::LinAlg::Vector<double>> inc) override;

    //! access to global (combined) increment of coupled problem
    std::shared_ptr<const Core::LinAlg::Vector<double>> combined_increment(
        const std::shared_ptr<const Core::LinAlg::Vector<double>> inc) const override;

    //! check if initial fields on coupled DOFs are equal
    void check_initial_fields(
        std::shared_ptr<const Core::LinAlg::Vector<double>> vec_cont) const override;

    //! set the element pairs that are close as found by search algorithm
    void set_nearby_ele_pairs(const std::map<int, std::set<int>>* nearbyelepairs) override;

    //! setup the strategy
    void setup() override;

    //! apply the mesh movement
    void apply_mesh_movement() const override;
  };

}  // namespace PoroPressureBased



FOUR_C_NAMESPACE_CLOSE

#endif
